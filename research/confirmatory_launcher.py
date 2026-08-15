"""Fail-closed launcher for the frozen Paper V1 confirmatory plan.

This module schedules existing frozen runners. It does not inspect outcome
metrics, change planned conditions, or silently repair failed attempts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from gym.model_config import normalize_provider
from research.planner import discover_manifests, load_plan, reconcile_plan
from research.run_artifacts import canonical_hash, file_set_hash


class LauncherError(RuntimeError):
    """Raised before provider access when the frozen launch contract drifts."""


@dataclass(frozen=True)
class FrozenStudy:
    plan: dict[str, Any]
    components: dict[str, Any]
    record: dict[str, Any]
    plan_path: Path
    components_path: Path
    record_path: Path


@dataclass(frozen=True)
class Preflight:
    execution_repo: Path
    datasets_root: Path
    runs_root: Path
    models_config: Path
    sandbox_image: str
    sandbox_image_id: str
    report: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LauncherError(f"Expected a JSON object: {path}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _docker_image_id(image: str) -> str:
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LauncherError(f"Frozen sandbox image is unavailable: {image}") from exc
    image_id = completed.stdout.strip()
    if not image_id.startswith("sha256:"):
        raise LauncherError(f"Docker returned an invalid image ID for {image}")
    return image_id


def _tagged_json(repo: Path, tag: str, relative_path: Path) -> dict[str, Any]:
    try:
        return json.loads(_git(repo, "show", f"{tag}:{relative_path.as_posix()}"))
    except (json.JSONDecodeError, OSError, subprocess.CalledProcessError) as exc:
        raise LauncherError(
            f"Cannot read frozen contract from {tag}:{relative_path.as_posix()}"
        ) from exc


def load_frozen_study(
    plan_path: str | Path,
    components_path: str | Path,
    record_path: str | Path,
) -> FrozenStudy:
    plan_source = Path(plan_path).resolve()
    components_source = Path(components_path).resolve()
    record_source = Path(record_path).resolve()
    plan = load_plan(plan_source)
    components = _read_json(components_source)
    record = _read_json(record_source)
    expected = {
        "plan_id": plan["plan_id"],
        "plan_hash": plan["plan_hash"],
        "protocol_hash": plan["protocol_hash"],
        "normalized_plan_config_hash": plan["config_hash"],
        "planned_conditions": len(plan["conditions"]),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise LauncherError(f"Freeze record field {key} does not match the plan")
    execution_commits = {
        item["condition"]["git_commit"] for item in plan["conditions"]
    }
    if execution_commits != {components.get("execution_git_commit")}:
        raise LauncherError("Plan conditions do not bind the component execution commit")
    if record.get("execution_git_commit") != components.get("execution_git_commit"):
        raise LauncherError("Freeze record and component execution commits differ")
    return FrozenStudy(
        plan=plan,
        components=components,
        record=record,
        plan_path=plan_source,
        components_path=components_source,
        record_path=record_source,
    )


def _dataset_hash(dataset_dir: Path) -> str:
    files = [dataset_dir / name for name in ("meta.json", "train.csv", "val.csv", "test.csv")]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise LauncherError("Missing frozen dataset files: " + ", ".join(missing))
    return file_set_hash(files)


def _validate_datasets(study: FrozenStudy, datasets_root: Path) -> None:
    expected: dict[str, str] = {}
    for item in study.plan["conditions"]:
        condition = item["condition"]
        dataset_id = str(condition["dataset_id"])
        dataset_hash = str(condition["dataset_hash"])
        if dataset_id in expected and expected[dataset_id] != dataset_hash:
            raise LauncherError(f"Plan contains multiple hashes for {dataset_id}")
        expected[dataset_id] = dataset_hash
    for dataset_id, dataset_hash in sorted(expected.items()):
        actual = _dataset_hash(datasets_root / dataset_id)
        if actual != dataset_hash:
            raise LauncherError(
                f"Frozen dataset hash mismatch for {dataset_id}: {actual} != {dataset_hash}"
            )


def _load_model_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise LauncherError("Model registry must be a JSON list")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _validate_models(study: FrozenStudy, models_config: Path) -> None:
    records = _load_model_records(models_config)
    by_name: dict[str, dict[str, Any]] = {}
    for item in records:
        name = str(item.get("name") or "")
        if name in by_name:
            raise LauncherError(f"Duplicate model registry name: {name!r}")
        by_name[name] = item
    frozen_env = study.components["runtime_environment"]
    required_ids = {
        str(item["condition"]["model_id"]) for item in study.plan["conditions"]
    }
    for model_id in sorted(required_ids):
        record = by_name.get(model_id)
        if record is None:
            raise LauncherError(f"Frozen model is absent from the registry: {model_id}")
        if normalize_provider(str(record.get("provider") or "")) != "openai":
            raise LauncherError(f"Frozen model {model_id} is not OpenAI-compatible")
        if str(record.get("baseUrl") or "") != frozen_env["LLM_BASE_URL"]:
            raise LauncherError(f"Frozen endpoint drift for model {model_id}")
        try:
            temperature_matches = float(record.get("temp")) == float(
                frozen_env["AUTOVIBE_LLM_TEMPERATURE"]
            )
            max_tokens_matches = int(record.get("maxTokens")) == int(
                study.components["decoding_config"]["payload"]["max_tokens"]
            )
        except (TypeError, ValueError) as exc:
            raise LauncherError(f"Invalid decoding config for model {model_id}") from exc
        if not temperature_matches:
            raise LauncherError(f"Frozen temperature drift for model {model_id}")
        if not max_tokens_matches:
            raise LauncherError(f"Frozen maxTokens drift for model {model_id}")
        if not str(record.get("apiKey") or "").strip():
            raise LauncherError(f"Model {model_id} has no API credential")


def _launcher_events(root: Path) -> list[dict[str, Any]]:
    path = root / "launcher_events.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LauncherError(f"Invalid launcher event at line {line_number}") from exc
        if not isinstance(event, dict):
            raise LauncherError(f"Launcher event {line_number} is not an object")
        events.append(event)
    return events


def _validate_launcher_history(
    study: FrozenStudy,
    *,
    root: Path,
    manifests: list[dict[str, Any]],
    sandbox_image: str,
    sandbox_image_id: str,
) -> None:
    events = _launcher_events(root)
    if manifests and not events:
        raise LauncherError("Run manifests exist without launcher provenance")
    by_run: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        run_id = str(event.get("run_id") or "")
        if not run_id:
            raise LauncherError("Launcher event is missing run_id")
        expected = {
            "plan_id": study.plan["plan_id"],
            "plan_hash": study.plan["plan_hash"],
            "execution_git_commit": study.record["execution_git_commit"],
            "sandbox_image": sandbox_image,
            "sandbox_image_id": sandbox_image_id,
        }
        if any(event.get(key) != value for key, value in expected.items()):
            raise LauncherError(f"Launcher provenance drift for {run_id}")
        by_run.setdefault(run_id, []).append(event)
    for run_id, run_events in by_run.items():
        phases = [event.get("phase") for event in run_events]
        if phases.count("started") != 1 or phases.count("finished") > 1:
            raise LauncherError(f"Invalid launcher lifecycle for {run_id}")
        if "finished" not in phases:
            raise LauncherError(f"Unfinished launcher lifecycle for {run_id}; human review required")
    manifest_ids = {str(manifest["run_id"]) for manifest in manifests}
    if manifest_ids != set(by_run):
        raise LauncherError("Launcher events and run manifests do not describe the same attempts")


def preflight(
    study: FrozenStudy,
    *,
    freeze_repo: str | Path,
    execution_repo: str | Path,
    datasets_root: str | Path,
    runs_root: str | Path,
    models_config: str | Path,
    sandbox_image: str,
) -> Preflight:
    freeze_root = Path(freeze_repo).resolve()
    execution_root = Path(execution_repo).resolve()
    datasets = Path(datasets_root).resolve()
    runs = Path(runs_root).resolve()
    registry = Path(models_config).resolve()
    expected_series_id = str(study.record.get("results_series_id") or "")
    if not expected_series_id:
        raise LauncherError("Freeze record does not bind a results series ID")
    if runs.parent.name != expected_series_id:
        raise LauncherError(
            f"Results series root must be named {expected_series_id}; got {runs.parent.name}"
        )
    tag = str(study.record["freeze_tag"])
    try:
        if _git(freeze_root, "cat-file", "-t", tag) != "tag":
            raise LauncherError(f"Freeze ref is not an annotated tag: {tag}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LauncherError(f"Cannot resolve immutable freeze tag: {tag}") from exc
    tagged_contract = {
        study.plan_path: study.plan,
        study.components_path: study.components,
        study.record_path: study.record,
    }
    repo_root = Path(__file__).resolve().parents[1]
    for local_path, local_value in tagged_contract.items():
        try:
            relative_path = local_path.relative_to(repo_root)
        except ValueError as exc:
            raise LauncherError(f"Frozen contract path is outside the repository: {local_path}") from exc
        if _tagged_json(freeze_root, tag, relative_path) != local_value:
            raise LauncherError(
                f"Local {relative_path.as_posix()} does not match the immutable freeze tag"
            )
    expected_execution = str(study.record["execution_git_commit"])
    if _git(execution_root, "rev-parse", "HEAD") != expected_execution:
        raise LauncherError("Execution worktree is not at the frozen execution commit")
    if _git(execution_root, "status", "--porcelain"):
        raise LauncherError("Execution worktree must be clean before every launch")
    _validate_datasets(study, datasets)
    _validate_models(study, registry)
    image_id = _docker_image_id(sandbox_image)
    manifests = discover_manifests(runs) if runs.exists() else []
    _validate_launcher_history(
        study,
        root=runs.parent,
        manifests=manifests,
        sandbox_image=sandbox_image,
        sandbox_image_id=image_id,
    )
    report = reconcile_plan(study.plan, manifests)
    if report["counts"]["running"]:
        raise LauncherError("A running manifest exists; recovery requires human review")
    return Preflight(
        execution_repo=execution_root,
        datasets_root=datasets,
        runs_root=runs,
        models_config=registry,
        sandbox_image=sandbox_image,
        sandbox_image_id=image_id,
        report=report,
    )


def _attempt_number(report: Mapping[str, Any], condition_id: str) -> int:
    status = next(
        item for item in report["conditions"] if item["condition_id"] == condition_id
    )
    return int(status["attempt_count"]) + 1


def _run_id(sequence_index: int, condition_id: str, attempt: int) -> str:
    return f"run_{sequence_index:03d}_{condition_id.removeprefix('cond_')}_attempt_{attempt:02d}"


def build_command(
    study: FrozenStudy,
    preflight_result: Preflight,
    action: Mapping[str, Any],
) -> tuple[list[str], str, Path]:
    planned = study.plan["conditions"][int(action["sequence_index"])]
    condition = planned["condition"]
    if planned["condition_id"] != action["condition_id"]:
        raise LauncherError("Runnable action does not match its planned sequence index")
    arm = str(condition["arm"])
    component = study.components["arms"][arm]
    attempt = _attempt_number(preflight_result.report, planned["condition_id"])
    run_id = _run_id(int(planned["sequence_index"]), planned["condition_id"], attempt)
    workspace = preflight_result.runs_root.parent / "workspaces" / run_id
    budget = study.components["budget_policy"]["payload"]
    command = [
        sys.executable,
        "-m",
        str(component["runner"]),
        "--dataset-dir",
        str(preflight_result.datasets_root / condition["dataset_id"]),
        "--model",
        str(condition["model_id"]),
        "--mode",
        "local",
        "--max-tokens",
        str(study.components["decoding_config"]["payload"]["max_tokens"]),
        "--sandbox-timeout",
        str(component["execution_policy"]["timeout_seconds"]),
        "--workspace-dir",
        str(workspace),
        "--research-run-dir",
        str(preflight_result.runs_root),
        "--research-experiment-id",
        str(condition["experiment_id"]),
        "--research-replicate-index",
        str(condition["replicate_index"]),
        "--research-arm",
        arm,
        "--research-model-version",
        str(condition["model_version"]),
        "--research-split-id",
        str(condition["split_id"]),
        "--research-condition-id",
        planned["condition_id"],
        "--research-run-id",
        run_id,
        "--research-total-token-limit",
        str(budget["total_token_limit"]),
        "--research-max-output-tokens",
        str(budget["max_output_tokens_per_call"]),
        "--research-max-llm-calls",
        str(budget["max_llm_calls"]),
        "--research-max-code-executions",
        str(budget["max_code_executions"]),
        "--research-max-tool-calls",
        str(budget["max_tool_calls"]),
        "--research-wall-clock-limit",
        str(budget["wall_clock_limit_seconds"]),
    ]
    if arm == "A":
        command.extend(["--executor-backend", "docker"])
    else:
        command.extend(["--episode-mode", str(component["product_mode"])])
    if action["action"] == "replace":
        command.extend(
            [
                "--research-rerun-of",
                str(action["rerun_of"]),
                "--research-rerun-reason",
                str(action["rerun_reason"]),
            ]
        )
    return command, run_id, workspace


def _runtime_env(study: FrozenStudy, preflight_result: Preflight) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {str(key): str(value) for key, value in study.components["runtime_environment"].items()}
    )
    env.update(
        {
            "AUTOVIBE_MODELS_CONFIG": str(preflight_result.models_config),
            "AUTOVIBE_SANDBOX_IMAGE": preflight_result.sandbox_image,
            "MLFLOW_TRACKING_URI": (
                "sqlite:///" + (preflight_result.runs_root.parent / "mlflow.db").as_posix()
            ),
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return env


def _write_launcher_record(
    study: FrozenStudy,
    preflight_result: Preflight,
    *,
    run_id: str,
    action: Mapping[str, Any],
    phase: str,
    return_code: int | None = None,
) -> None:
    root = preflight_result.runs_root.parent
    record_path = root / "launcher_events.jsonl"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": _utc_now(),
        "plan_id": study.plan["plan_id"],
        "plan_hash": study.plan["plan_hash"],
        "execution_git_commit": study.record["execution_git_commit"],
        "sandbox_image": preflight_result.sandbox_image,
        "sandbox_image_id": preflight_result.sandbox_image_id,
        "sequence_index": int(action["sequence_index"]),
        "condition_id": str(action["condition_id"]),
        "action": str(action["action"]),
        "run_id": run_id,
        "phase": phase,
    }
    if return_code is not None:
        event["return_code"] = int(return_code)
    with record_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_next(
    study: FrozenStudy,
    preflight_result: Preflight,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    if preflight_result.report["complete"]:
        return preflight_result.report
    action = preflight_result.report["runnable"][0]
    command, run_id, workspace = build_command(study, preflight_result, action)
    if dry_run:
        print(
            json.dumps(
                {
                    "action": action,
                    "run_id": run_id,
                    "runner": command[2],
                    "sandbox_image": preflight_result.sandbox_image,
                    "sandbox_image_id": preflight_result.sandbox_image_id,
                },
                indent=2,
            )
        )
        return preflight_result.report
    logs = preflight_result.runs_root.parent / "launcher_logs"
    logs.mkdir(parents=True, exist_ok=True)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    preflight_result.runs_root.mkdir(parents=True, exist_ok=True)
    with (logs / f"{run_id}.stdout.log").open("x", encoding="utf-8") as stdout, (
        logs / f"{run_id}.stderr.log"
    ).open("x", encoding="utf-8") as stderr:
        _write_launcher_record(
            study,
            preflight_result,
            run_id=run_id,
            action=action,
            phase="started",
        )
        completed = subprocess.run(
            command,
            cwd=preflight_result.execution_repo,
            env=_runtime_env(study, preflight_result),
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    _write_launcher_record(
        study,
        preflight_result,
        run_id=run_id,
        action=action,
        phase="finished",
        return_code=completed.returncode,
    )
    report = reconcile_plan(study.plan, discover_manifests(preflight_result.runs_root))
    status = next(
        item for item in report["conditions"] if item["condition_id"] == action["condition_id"]
    )
    if status["state"] == "running":
        raise LauncherError(f"Runner left an incomplete manifest for {run_id}")
    if completed.returncode != 0 and status["state"] != "replacement_required":
        raise LauncherError(
            f"Runner exited {completed.returncode}; inspect infrastructure logs for {run_id}"
        )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "sequence_index": action["sequence_index"],
                "condition_id": action["condition_id"],
                "state": status["state"],
                "counts": report["counts"],
            },
            indent=2,
        )
    )
    return report


def main() -> None:
    protocols = Path(__file__).resolve().parent / "protocols"
    parser = argparse.ArgumentParser(description="Launch the frozen confirmatory plan")
    parser.add_argument("--plan", default=str(protocols / "confirmatory_plan_v1.json"))
    parser.add_argument(
        "--components", default=str(protocols / "confirmatory_components_v1.json")
    )
    parser.add_argument("--freeze-record", default=str(protocols / "freeze_record_v1.json"))
    parser.add_argument("--freeze-repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--execution-repo", required=True)
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--sandbox-image", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-conditions", type=int, default=1)
    args = parser.parse_args()
    if args.max_conditions <= 0:
        raise LauncherError("--max-conditions must be positive")
    study = load_frozen_study(args.plan, args.components, args.freeze_record)
    completed = 0
    while completed < args.max_conditions:
        ready = preflight(
            study,
            freeze_repo=args.freeze_repo,
            execution_repo=args.execution_repo,
            datasets_root=args.datasets_root,
            runs_root=args.runs_root,
            models_config=args.models_config,
            sandbox_image=args.sandbox_image,
        )
        if ready.report["complete"]:
            print(json.dumps({"complete": True, "counts": ready.report["counts"]}, indent=2))
            return
        report = run_next(study, ready, dry_run=args.dry_run)
        if args.dry_run:
            return
        completed += 1
        if report["counts"]["replacement_required"]:
            raise LauncherError(
                "Infrastructure replacement is queued; stop for human review before resume"
            )


if __name__ == "__main__":
    main()

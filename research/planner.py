"""Deterministic Paper V1 confirmatory matrix planning and resume checks."""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from research.run_artifacts import (
    REQUIRED_CONDITION_FIELDS,
    FailureCategory,
    canonical_hash,
    condition_id,
    validate_manifest,
)


PLAN_SCHEMA_VERSION = "1.0"
PLAN_KIND = "confirmatory_experiment_plan"
ARM_IDS = ("A", "B", "C")
BLOCK_KEYS = ("dataset_id", "model_id", "replicate_index")
INFRASTRUCTURE_FAILURES = {
    FailureCategory.SANDBOX_FAILURE.value,
    FailureCategory.PROVIDER_RATE_LIMIT.value,
    FailureCategory.PROVIDER_CAPACITY.value,
    FailureCategory.PROVIDER_AUTH.value,
    FailureCategory.PROVIDER_OTHER.value,
    FailureCategory.ORCHESTRATOR_FAILURE.value,
}
_HASH = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_PLACEHOLDER = re.compile(
    r"(^|[^a-z])(todo|tbd|pending|unknown|unversioned)([^a-z]|$)",
    re.IGNORECASE,
)


class PlanError(ValueError):
    """Raised when a plan or run set violates the research contract."""


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanError(f"{label} must be a mapping")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise PlanError(f"{label} must be a non-empty list")
    return value


def _require_hash(value: Any, label: str) -> str:
    text = str(value or "")
    if not _HASH.fullmatch(text):
        raise PlanError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _reject_placeholders(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_placeholders(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_placeholders(item, f"{path}[{index}]")
        return
    if value is None:
        raise PlanError(f"{path} must be resolved before planning")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized or normalized in {"unknown", "unversioned", "pending"}:
            raise PlanError(f"{path} contains unresolved value {value!r}")
        if _PLACEHOLDER.search(normalized):
            raise PlanError(f"{path} contains unresolved placeholder {value!r}")


def _unique(items: list[Mapping[str, Any]], key: str, label: str) -> None:
    values = [str(item.get(key) or "") for item in items]
    if any(not value for value in values):
        raise PlanError(f"Every {label} entry must define {key}")
    if len(values) != len(set(values)):
        raise PlanError(f"Duplicate {key} in {label}")


def _arm_order(seed: int, block: Mapping[str, Any], arms: list[str]) -> list[str]:
    """Return a portable seeded permutation without runtime RNG dependencies."""

    return sorted(
        arms,
        key=lambda arm: canonical_hash(
            {"randomization_seed": seed, "block": dict(block), "arm": arm}
        ),
    )


def _plan_digest(plan: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in plan.items() if key not in {"plan_id", "plan_hash"}}
    return canonical_hash(payload)


def _validate_protocol_contract(config: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    protocol_matrix = _require_mapping(protocol.get("matrix"), "protocol.matrix")
    protocol_arms = _require_mapping(protocol.get("arms"), "protocol.arms")
    protocol_budget = _require_mapping(protocol.get("budget"), "protocol.budget")
    _reject_placeholders(protocol_matrix, "protocol.matrix")
    _reject_placeholders(
        {arm_id: protocol_arms.get(arm_id) for arm_id in ARM_IDS},
        "protocol.arms",
    )
    budget_status = str(protocol_budget.get("status") or "").strip().lower()
    if not budget_status or "pilot" in budget_status or "not_frozen" in budget_status:
        raise PlanError("protocol.budget.status must identify the final frozen budget")

    matrix = _require_mapping(config.get("matrix"), "config.matrix")
    protocol_model_ids = sorted(str(item.get("id") or "") for item in protocol_matrix.get("models", []))
    config_model_ids = sorted(str(item.get("model_id") or "") for item in matrix.get("models", []))
    protocol_dataset_ids = sorted(str(item.get("id") or "") for item in protocol_matrix.get("datasets", []))
    config_dataset_ids = sorted(str(item.get("dataset_id") or "") for item in matrix.get("datasets", []))
    if protocol_model_ids != config_model_ids:
        raise PlanError("config model IDs do not match protocol.matrix.models")
    if protocol_dataset_ids != config_dataset_ids:
        raise PlanError("config dataset IDs do not match protocol.matrix.datasets")
    if protocol_matrix.get("confirmatory_arms") != list(ARM_IDS):
        raise PlanError("protocol.matrix.confirmatory_arms must be ordered A, B, C")
    comparisons = {
        "replicates_per_cell": "replicates_per_cell",
        "planned_confirmatory_runs": "planned_conditions",
        "randomization_seed": "randomization_seed",
    }
    for protocol_key, config_key in comparisons.items():
        if protocol_matrix.get(protocol_key) != matrix.get(config_key):
            raise PlanError(f"config.matrix.{config_key} does not match protocol.matrix.{protocol_key}")
    if protocol_matrix.get("block_keys") != list(BLOCK_KEYS):
        raise PlanError("protocol.matrix.block_keys do not match the planner contract")
    for arm in matrix.get("arms", []):
        arm_id = str(arm.get("arm") or "")
        expected_mode = _require_mapping(protocol_arms.get(arm_id), f"protocol.arms.{arm_id}").get(
            "current_product_mode"
        )
        if arm.get("product_mode") != expected_mode:
            raise PlanError(f"config product mode for arm {arm_id} does not match protocol.arms")


def build_confirmatory_plan(
    config: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand an exact, result-blind config into an immutable condition plan."""

    config = dict(_require_mapping(config, "config"))
    protocol = dict(_require_mapping(protocol, "protocol"))
    _reject_placeholders(config)

    if config.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise PlanError(f"config.schema_version must be {PLAN_SCHEMA_VERSION!r}")
    experiment_id = str(config.get("experiment_id") or "")
    git_commit = str(config.get("git_commit") or "")
    if not experiment_id:
        raise PlanError("config.experiment_id is required")
    if not _GIT_COMMIT.fullmatch(git_commit):
        raise PlanError("config.git_commit must be a lowercase 40-character Git commit")

    matrix = _require_mapping(config.get("matrix"), "config.matrix")
    datasets = [dict(_require_mapping(item, "dataset")) for item in _require_list(matrix.get("datasets"), "config.matrix.datasets")]
    models = [dict(_require_mapping(item, "model")) for item in _require_list(matrix.get("models"), "config.matrix.models")]
    arms = [dict(_require_mapping(item, "arm")) for item in _require_list(matrix.get("arms"), "config.matrix.arms")]
    _unique(datasets, "dataset_id", "datasets")
    _unique(models, "model_id", "models")
    _unique(arms, "arm", "arms")
    arm_ids = {str(item["arm"]) for item in arms}
    if arm_ids != set(ARM_IDS):
        raise PlanError("config.matrix.arms must contain exactly A, B, and C")

    replicates = int(matrix.get("replicates_per_cell") or 0)
    seed = int(matrix.get("randomization_seed") or 0)
    planned_conditions = int(matrix.get("planned_conditions") or 0)
    if replicates <= 0:
        raise PlanError("config.matrix.replicates_per_cell must be positive")
    expected_conditions = len(datasets) * len(models) * len(arms) * replicates
    if planned_conditions != expected_conditions:
        raise PlanError(
            "config.matrix.planned_conditions must equal datasets * models * arms * replicates"
        )
    _validate_protocol_contract(config, protocol)

    budget_policy_hash = _require_hash(config.get("budget_policy_hash"), "config.budget_policy_hash")
    for item in datasets:
        _require_hash(item.get("dataset_hash"), f"dataset {item.get('dataset_id')} dataset_hash")
        if int(item.get("split_seed", -1)) < 0:
            raise PlanError(f"dataset {item.get('dataset_id')} split_seed must be non-negative")
    for item in models:
        _require_hash(item.get("decoding_config_hash"), f"model {item.get('model_id')} decoding_config_hash")
    for item in arms:
        _require_hash(item.get("execution_policy_hash"), f"arm {item.get('arm')} execution_policy_hash")
        _require_hash(item.get("prompt_template_hash"), f"arm {item.get('arm')} prompt_template_hash")

    datasets.sort(key=lambda item: str(item["dataset_id"]))
    models.sort(key=lambda item: str(item["model_id"]))
    arms_by_id = {str(item["arm"]): item for item in arms}
    normalized_matrix = dict(matrix)
    normalized_matrix.update(
        {
            "datasets": datasets,
            "models": models,
            "arms": [arms_by_id[arm_id] for arm_id in ARM_IDS],
        }
    )
    normalized_config = dict(config)
    normalized_config["matrix"] = normalized_matrix
    conditions: list[dict[str, Any]] = []
    block_index = 0
    sequence_index = 0
    for dataset in datasets:
        for model in models:
            for replicate_index in range(replicates):
                block = {
                    "dataset_id": dataset["dataset_id"],
                    "model_id": model["model_id"],
                    "replicate_index": replicate_index,
                }
                block_id = f"block_{canonical_hash(block)[:24]}"
                order = _arm_order(seed, block, list(ARM_IDS))
                for within_block_index, arm_id in enumerate(order):
                    arm = arms_by_id[arm_id]
                    condition = {
                        "experiment_id": experiment_id,
                        "git_commit": git_commit,
                        "dataset_id": dataset["dataset_id"],
                        "dataset_hash": dataset["dataset_hash"],
                        "split_id": dataset["split_id"],
                        "split_seed": int(dataset["split_seed"]),
                        "model_provider": model["model_provider"],
                        "model_id": model["model_id"],
                        "model_version": model["model_version"],
                        "decoding_config_hash": model["decoding_config_hash"],
                        "arm": arm_id,
                        "replicate_index": replicate_index,
                        "budget_policy_hash": budget_policy_hash,
                        "execution_policy_hash": arm["execution_policy_hash"],
                        "prompt_template_hash": arm["prompt_template_hash"],
                    }
                    conditions.append(
                        {
                            "sequence_index": sequence_index,
                            "block_index": block_index,
                            "within_block_index": within_block_index,
                            "block_id": block_id,
                            "product_mode": arm["product_mode"],
                            "condition_id": condition_id(condition),
                            "condition": condition,
                        }
                    )
                    sequence_index += 1
                block_index += 1

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "experiment_id": experiment_id,
        "protocol_hash": canonical_hash(protocol),
        "config_hash": canonical_hash(normalized_config),
        "matrix": {
            "dataset_ids": [str(item["dataset_id"]) for item in datasets],
            "model_ids": [str(item["model_id"]) for item in models],
            "arms": list(ARM_IDS),
            "replicates_per_cell": replicates,
            "block_keys": list(BLOCK_KEYS),
            "block_count": block_index,
            "planned_conditions": planned_conditions,
            "randomization_seed": seed,
            "within_block_order": "sha256_seeded_permutation_of_A_B_C",
        },
        "conditions": conditions,
    }
    plan_hash = _plan_digest(plan)
    plan["plan_id"] = f"plan_{plan_hash[:24]}"
    plan["plan_hash"] = plan_hash
    validate_plan(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "kind",
        "experiment_id",
        "protocol_hash",
        "config_hash",
        "plan_id",
        "plan_hash",
        "matrix",
        "conditions",
    }
    missing = sorted(required - set(plan))
    if missing:
        raise PlanError(f"Plan is missing required fields: {', '.join(missing)}")
    if plan["schema_version"] != PLAN_SCHEMA_VERSION or plan["kind"] != PLAN_KIND:
        raise PlanError("Unsupported confirmatory plan schema or kind")
    _require_hash(plan["protocol_hash"], "plan.protocol_hash")
    _require_hash(plan["config_hash"], "plan.config_hash")
    expected_hash = _plan_digest(plan)
    if plan["plan_hash"] != expected_hash:
        raise PlanError("plan_hash does not match the canonical plan payload")
    if plan["plan_id"] != f"plan_{expected_hash[:24]}":
        raise PlanError("plan_id does not match plan_hash")

    matrix = _require_mapping(plan["matrix"], "plan.matrix")
    conditions = _require_list(plan["conditions"], "plan.conditions")
    if int(matrix.get("planned_conditions") or -1) != len(conditions):
        raise PlanError("Plan condition count does not match matrix.planned_conditions")
    if matrix.get("block_keys") != list(BLOCK_KEYS):
        raise PlanError("Plan block_keys do not match the Paper V1 protocol")
    if matrix.get("arms") != list(ARM_IDS):
        raise PlanError("Plan arms must be ordered A, B, C")

    condition_ids: set[str] = set()
    blocks: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for expected_index, raw_item in enumerate(conditions):
        item = _require_mapping(raw_item, f"plan.conditions[{expected_index}]")
        if item.get("sequence_index") != expected_index:
            raise PlanError("Plan sequence_index values must be contiguous and ordered")
        condition = _require_mapping(item.get("condition"), "planned condition")
        computed_id = condition_id(condition)
        if item.get("condition_id") != computed_id:
            raise PlanError("Planned condition_id does not match its condition payload")
        if computed_id in condition_ids:
            raise PlanError(f"Duplicate planned condition_id: {computed_id}")
        condition_ids.add(computed_id)
        blocks[str(item.get("block_id"))].append(item)

    if len(blocks) != int(matrix.get("block_count") or -1):
        raise PlanError("Plan block count does not match matrix.block_count")
    for block_id, items in blocks.items():
        arms = {str(item["condition"]["arm"]) for item in items}
        positions = {int(item["within_block_index"]) for item in items}
        if arms != set(ARM_IDS) or positions != {0, 1, 2}:
            raise PlanError(f"Block {block_id} must contain A/B/C exactly once")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_plan(plan: Mapping[str, Any], path: str | Path) -> bool:
    """Atomically create a plan; identical recreation is an idempotent no-op."""

    validate_plan(plan)
    destination = Path(path)
    content = _json_bytes(plan)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        validate_plan(existing)
        if existing.get("plan_hash") != plan.get("plan_hash") or _json_bytes(existing) != content:
            raise FileExistsError(f"Refusing to overwrite different plan: {destination}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def load_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_plan(plan)
    return plan


def discover_manifests(root: str | Path) -> list[dict[str, Any]]:
    manifests = []
    for path in sorted(Path(root).rglob("run_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        manifest = dict(manifest)
        manifest["_manifest_path"] = str(path.resolve())
        manifests.append(manifest)
    return manifests


def _condition_fields(condition: Mapping[str, Any]) -> dict[str, Any]:
    return {key: condition[key] for key in REQUIRED_CONDITION_FIELDS}


def reconcile_plan(
    plan: Mapping[str, Any], manifests: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Reconcile immutable planned conditions with append-only run attempts."""

    validate_plan(plan)
    planned_items = {item["condition_id"]: item for item in plan["conditions"]}
    attempts_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_ids: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for raw_manifest in manifests:
        manifest = dict(raw_manifest)
        validate_manifest(manifest)
        run_id = str(manifest["run_id"])
        condition_key = str(manifest["condition_id"])
        if run_id in run_ids:
            errors.append(f"duplicate run_id {run_id}")
            continue
        run_ids[run_id] = manifest
        planned = planned_items.get(condition_key)
        if planned is None:
            errors.append(f"run {run_id} has unplanned condition_id {condition_key}")
            continue
        if manifest.get("experiment_id") != plan.get("experiment_id"):
            errors.append(f"run {run_id} has wrong experiment_id")
        if _condition_fields(manifest["condition"]) != _condition_fields(planned["condition"]):
            errors.append(f"run {run_id} condition payload differs from the immutable plan")
        attempts_by_condition[condition_key].append(manifest)

    statuses: list[dict[str, Any]] = []
    replacement_queue: list[dict[str, Any]] = []
    runnable: list[dict[str, Any]] = []
    counts = {"pending": 0, "running": 0, "terminal": 0, "replacement_required": 0}

    for item in plan["conditions"]:
        condition_key = item["condition_id"]
        attempts = attempts_by_condition.get(condition_key, [])
        if not attempts:
            state = "pending"
            runnable.append(
                {"action": "start", "condition_id": condition_key, "sequence_index": item["sequence_index"]}
            )
            latest_run_id = None
        else:
            by_id = {str(attempt["run_id"]): attempt for attempt in attempts}
            children: dict[str, list[str]] = defaultdict(list)
            roots: list[str] = []
            for attempt in attempts:
                run_id = str(attempt["run_id"])
                parent_id = attempt.get("rerun_of")
                if parent_id is None:
                    roots.append(run_id)
                    continue
                parent = by_id.get(str(parent_id))
                if parent is None:
                    errors.append(f"replacement {run_id} references missing run {parent_id}")
                    continue
                children[str(parent_id)].append(run_id)
                if parent.get("state") != "completed" or parent.get("failure_category") not in INFRASTRUCTURE_FAILURES:
                    errors.append(f"replacement {run_id} does not follow a completed infrastructure failure")
                if not attempt.get("rerun_reason"):
                    errors.append(f"replacement {run_id} is missing rerun_reason")
            if len(roots) != 1:
                errors.append(f"condition {condition_key} must have exactly one original attempt")
            for parent_id, child_ids in children.items():
                if len(child_ids) > 1:
                    errors.append(f"run {parent_id} has multiple replacement children")

            chain: list[dict[str, Any]] = []
            if len(roots) == 1:
                cursor = roots[0]
                seen: set[str] = set()
                while cursor not in seen:
                    seen.add(cursor)
                    chain.append(by_id[cursor])
                    next_ids = children.get(cursor, [])
                    if len(next_ids) != 1:
                        break
                    cursor = next_ids[0]
                if len(chain) != len(attempts):
                    errors.append(f"condition {condition_key} attempts do not form one replacement chain")
            if not chain:
                chain = sorted(attempts, key=lambda attempt: (str(attempt.get("started_at") or ""), str(attempt["run_id"])))

            running = [attempt for attempt in chain if attempt.get("state") == "running"]
            terminal = [
                attempt
                for attempt in chain
                if attempt.get("state") == "completed"
                and attempt.get("failure_category") not in INFRASTRUCTURE_FAILURES
            ]
            if len(running) > 1 or (running and running[-1] is not chain[-1]):
                errors.append(f"condition {condition_key} has conflicting running attempts")
            if len(terminal) > 1:
                errors.append(f"condition {condition_key} has duplicate terminal agent outcomes")
            latest = chain[-1]
            latest_run_id = str(latest["run_id"])
            if terminal:
                state = "terminal"
            elif running:
                state = "running"
            elif latest.get("failure_category") in INFRASTRUCTURE_FAILURES:
                state = "replacement_required"
                queue_item = {
                    "action": "replace",
                    "condition_id": condition_key,
                    "sequence_index": item["sequence_index"],
                    "rerun_of": latest_run_id,
                    "rerun_reason": f"replacement after {latest.get('failure_category')}",
                }
                replacement_queue.append(queue_item)
                runnable.append(queue_item)
            else:
                errors.append(f"condition {condition_key} has an invalid latest attempt state")
                state = "terminal"

        counts[state] += 1
        statuses.append(
            {
                "sequence_index": item["sequence_index"],
                "condition_id": condition_key,
                "state": state,
                "latest_run_id": latest_run_id,
                "attempt_count": len(attempts),
            }
        )

    if errors:
        raise PlanError("Run-set reconciliation failed:\n- " + "\n- ".join(sorted(set(errors))))
    report: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "plan_hash": plan["plan_hash"],
        "counts": counts,
        "complete": counts["terminal"] == len(plan["conditions"]),
        "conditions": statuses,
        "replacement_queue": replacement_queue,
        "runnable": sorted(runnable, key=lambda item: int(item["sequence_index"])),
    }
    report["reconciliation_hash"] = canonical_hash(report)
    return report


def _load_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if source.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    return dict(_require_mapping(value, str(source)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper V1 confirmatory experiment planner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build an immutable condition plan")
    build_parser.add_argument("--config", required=True)
    build_parser.add_argument(
        "--protocol", default=str(Path(__file__).parent / "protocols" / "protocol_v1.yaml")
    )
    build_parser.add_argument("--output", required=True)

    status_parser = subparsers.add_parser("status", help="Reconcile a plan with run manifests")
    status_parser.add_argument("--plan", required=True)
    status_parser.add_argument("--runs", required=True)
    status_parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.command == "build":
        plan = build_confirmatory_plan(
            _load_mapping(args.config), protocol=_load_mapping(args.protocol)
        )
        created = write_plan(plan, args.output)
        print(
            json.dumps(
                {
                    "plan_id": plan["plan_id"],
                    "plan_hash": plan["plan_hash"],
                    "planned_conditions": len(plan["conditions"]),
                    "created": created,
                    "output": str(Path(args.output).resolve()),
                },
                indent=2,
            )
        )
        return

    report = reconcile_plan(load_plan(args.plan), discover_manifests(args.runs))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_json_bytes(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

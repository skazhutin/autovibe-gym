import json
from pathlib import Path

import pytest

import research.confirmatory_launcher as launcher
from research.confirmatory_launcher import (
    FrozenStudy,
    LauncherError,
    Preflight,
    build_command,
    load_frozen_study,
    preflight,
)
from research.planner import reconcile_plan


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "research" / "protocols"


def _study():
    return load_frozen_study(
        PROTOCOLS / "confirmatory_plan_v1.json",
        PROTOCOLS / "confirmatory_components_v1.json",
        PROTOCOLS / "freeze_record_v1.json",
    )


def _ready(study, tmp_path):
    return Preflight(
        execution_repo=tmp_path / "execution",
        datasets_root=tmp_path / "datasets",
        runs_root=tmp_path / "results" / "runs",
        models_config=tmp_path / "models.json",
        sandbox_image="autovibe-gym-sandbox:paper-v1",
        sandbox_image_id="sha256:" + "a" * 64,
        report=reconcile_plan(study.plan, []),
    )


def _action_for_arm(study, arm):
    planned = next(
        item for item in study.plan["conditions"] if item["condition"]["arm"] == arm
    )
    return {
        "action": "start",
        "condition_id": planned["condition_id"],
        "sequence_index": planned["sequence_index"],
    }


@pytest.mark.parametrize(
    ("arm", "module", "mode_flag"),
    [
        ("A", "experiments.run_multishot", "--executor-backend"),
        ("B", "experiments.run_gym", "iterative_no_checklist"),
        ("C", "experiments.run_gym", "gym_with_checklist"),
    ],
)
def test_commands_bind_each_frozen_arm_without_credentials(
    tmp_path, arm, module, mode_flag
):
    study = _study()
    command, run_id, workspace = build_command(
        study, _ready(study, tmp_path), _action_for_arm(study, arm)
    )

    assert command[1:3] == ["-m", module]
    assert "--research-condition-id" in command
    assert command[command.index("--research-arm") + 1] == arm
    assert mode_flag in command
    assert "LLM_API_KEY" not in " ".join(command)
    assert run_id.startswith("run_")
    assert workspace.name == run_id


def test_replacement_command_preserves_chain_fields(tmp_path):
    study = _study()
    ready = _ready(study, tmp_path)
    action = _action_for_arm(study, "A")
    action.update(
        {
            "action": "replace",
            "rerun_of": "run_original",
            "rerun_reason": "replacement after provider_capacity",
        }
    )

    command, _, _ = build_command(study, ready, action)

    assert command[command.index("--research-rerun-of") + 1] == "run_original"
    assert (
        command[command.index("--research-rerun-reason") + 1]
        == "replacement after provider_capacity"
    )


def test_freeze_record_drift_is_rejected(tmp_path):
    record = json.loads((PROTOCOLS / "freeze_record_v1.json").read_text("utf-8"))
    record["plan_hash"] = "0" * 64
    changed = tmp_path / "freeze.json"
    changed.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(LauncherError, match="plan_hash"):
        load_frozen_study(
            PROTOCOLS / "confirmatory_plan_v1.json",
            PROTOCOLS / "confirmatory_components_v1.json",
            changed,
        )


def test_preflight_rejects_dirty_execution_worktree(tmp_path, monkeypatch):
    study = _study()

    def fake_git(_repo, *args):
        if args == ("cat-file", "-t", study.record["freeze_tag"]):
            return "tag"
        if args[0] == "show":
            path = args[1].split(":", 1)[1]
            values = {
                "research/protocols/confirmatory_plan_v1.json": study.plan,
                "research/protocols/confirmatory_components_v1.json": study.components,
                "research/protocols/freeze_record_v1.json": study.record,
            }
            return json.dumps(values[path])
        if args == ("rev-parse", "HEAD"):
            return study.record["execution_git_commit"]
        if args == ("status", "--porcelain"):
            return "?? unexpected.txt"
        return study.record["execution_git_commit"]

    monkeypatch.setattr(launcher, "_git", fake_git)
    monkeypatch.setattr(launcher, "_validate_datasets", lambda *_args: None)
    monkeypatch.setattr(launcher, "_validate_models", lambda *_args: None)
    monkeypatch.setattr(launcher, "_docker_image_id", lambda _image: "sha256:" + "a" * 64)

    with pytest.raises(LauncherError, match="clean"):
        preflight(
            study,
            freeze_repo=tmp_path,
            execution_repo=tmp_path,
            datasets_root=tmp_path,
            runs_root=tmp_path / "runs",
            models_config=tmp_path / "models.json",
            sandbox_image="fixture",
        )


def test_preflight_accepts_exact_clean_contract(tmp_path, monkeypatch):
    study = _study()

    def fake_git(_repo, *args):
        if args == ("cat-file", "-t", study.record["freeze_tag"]):
            return "tag"
        if args[0] == "show":
            path = args[1].split(":", 1)[1]
            values = {
                "research/protocols/confirmatory_plan_v1.json": study.plan,
                "research/protocols/confirmatory_components_v1.json": study.components,
                "research/protocols/freeze_record_v1.json": study.record,
            }
            return json.dumps(values[path])
        if args == ("rev-parse", "HEAD"):
            return study.record["execution_git_commit"]
        if args == ("status", "--porcelain"):
            return ""
        return study.record["execution_git_commit"]

    monkeypatch.setattr(launcher, "_git", fake_git)
    monkeypatch.setattr(launcher, "_validate_datasets", lambda *_args: None)
    monkeypatch.setattr(launcher, "_validate_models", lambda *_args: None)
    monkeypatch.setattr(launcher, "_docker_image_id", lambda _image: "sha256:" + "a" * 64)

    ready = preflight(
        study,
        freeze_repo=tmp_path,
        execution_repo=tmp_path,
        datasets_root=tmp_path,
        runs_root=tmp_path / "runs",
        models_config=tmp_path / "models.json",
        sandbox_image="fixture",
    )

    assert ready.report["counts"]["pending"] == 120
    assert ready.report["counts"]["terminal"] == 0


def test_launcher_history_rejects_image_drift(tmp_path):
    study = _study()
    root = tmp_path / "results"
    root.mkdir()
    event = {
        "plan_id": study.plan["plan_id"],
        "plan_hash": study.plan["plan_hash"],
        "execution_git_commit": study.record["execution_git_commit"],
        "sandbox_image": "frozen-image",
        "sandbox_image_id": "sha256:" + "a" * 64,
        "run_id": "run_fixture",
        "phase": "started",
    }
    (root / "launcher_events.jsonl").write_text(json.dumps(event) + "\n", "utf-8")

    with pytest.raises(LauncherError, match="provenance drift"):
        launcher._validate_launcher_history(
            study,
            root=root,
            manifests=[],
            sandbox_image="frozen-image",
            sandbox_image_id="sha256:" + "b" * 64,
        )


def test_launcher_history_rejects_unfinished_attempt(tmp_path):
    study = _study()
    root = tmp_path / "results"
    root.mkdir()
    event = {
        "plan_id": study.plan["plan_id"],
        "plan_hash": study.plan["plan_hash"],
        "execution_git_commit": study.record["execution_git_commit"],
        "sandbox_image": "frozen-image",
        "sandbox_image_id": "sha256:" + "a" * 64,
        "run_id": "run_fixture",
        "phase": "started",
    }
    (root / "launcher_events.jsonl").write_text(json.dumps(event) + "\n", "utf-8")

    with pytest.raises(LauncherError, match="Unfinished launcher lifecycle"):
        launcher._validate_launcher_history(
            study,
            root=root,
            manifests=[],
            sandbox_image="frozen-image",
            sandbox_image_id="sha256:" + "a" * 64,
        )

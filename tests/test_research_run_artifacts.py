import json
from dataclasses import dataclass

import pytest

from research.run_artifacts import (
    FailureCategory,
    RecordingExecutor,
    RecordingLLMClient,
    RunRecorder,
    canonical_hash,
    classify_failure,
    condition_id,
    validate_manifest,
)


def _condition(**overrides):
    value = {
        "experiment_id": "paper-v1-offline-smoke",
        "git_commit": "a" * 40,
        "dataset_id": "fixture",
        "dataset_hash": "b" * 64,
        "split_id": "fixture-split-v1",
        "split_seed": 42,
        "model_provider": "fixture",
        "model_id": "fixture-model",
        "model_version": "fixture-v1",
        "decoding_config_hash": canonical_hash({"max_tokens": 32}),
        "arm": "single_shot",
        "replicate_index": 0,
        "budget_policy_hash": canonical_hash({"calls": 1}),
        "execution_policy_hash": canonical_hash({"backend": "fixture"}),
        "prompt_template_hash": canonical_hash({"prompt": "fixture"}),
    }
    value.update(overrides)
    return value


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_condition_id_is_stable_and_sensitive_to_condition():
    condition = _condition()
    reversed_condition = dict(reversed(list(condition.items())))

    assert condition_id(condition) == condition_id(reversed_condition)
    assert condition_id(condition) != condition_id(_condition(replicate_index=1))


def test_condition_requires_complete_non_empty_identity():
    missing = _condition()
    missing.pop("dataset_hash")
    with pytest.raises(ValueError, match="dataset_hash"):
        condition_id(missing)

    with pytest.raises(ValueError, match="model_version"):
        condition_id(_condition(model_version=""))


def test_run_ids_are_unique_and_explicit_collisions_do_not_overwrite(tmp_path):
    first = RunRecorder.create(tmp_path, _condition())
    second = RunRecorder.create(tmp_path, _condition())
    first.finalize({"final_status": "no_candidate_found", "valid_submit": False})
    second.finalize({"final_status": "no_candidate_found", "valid_submit": False})

    assert first.run_id != second.run_id
    with pytest.raises(FileExistsError):
        RunRecorder.create(tmp_path, _condition(), run_id=first.run_id)
    with pytest.raises(ValueError, match="run_id"):
        RunRecorder.create(tmp_path, _condition(), run_id="../outside")


def test_expected_condition_id_must_match(tmp_path):
    with pytest.raises(ValueError, match="Condition ID mismatch"):
        RunRecorder.create(tmp_path, _condition(), expected_condition_id="cond_deadbeef")


def test_rerun_requires_reason_and_links_original(tmp_path):
    with pytest.raises(ValueError, match="rerun_reason"):
        RunRecorder.create(tmp_path, _condition(), rerun_of="run_original")

    recorder = RunRecorder.create(
        tmp_path,
        _condition(),
        rerun_of="run_original",
        rerun_reason="provider outage approved by protocol",
    )
    recorder.finalize({"final_status": "submitted_clean", "valid_submit": True})
    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    assert manifest["rerun_of"] == "run_original"
    assert manifest["rerun_reason"] == "provider outage approved by protocol"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"valid_submit": True}, FailureCategory.SUCCESS),
        ({"final_status": "no_candidate_found"}, FailureCategory.INVALID_SUBMISSION),
        ({"final_status": "hidden_submit_failed"}, FailureCategory.INVALID_SUBMISSION),
        ({"final_status": "max_steps_exhausted"}, FailureCategory.BUDGET_EXHAUSTED),
        ({"error_type": "TimeoutError"}, FailureCategory.EXECUTION_TIMEOUT),
        ({"http_status": 429, "origin": "provider"}, FailureCategory.PROVIDER_RATE_LIMIT),
        ({"http_status": 401, "origin": "provider"}, FailureCategory.PROVIDER_AUTH),
        ({"http_status": 503, "origin": "provider"}, FailureCategory.PROVIDER_CAPACITY),
        ({"error_type": "UnknownAPIError", "origin": "provider"}, FailureCategory.PROVIDER_OTHER),
        ({"error_type": "SandboxCrash", "origin": "sandbox"}, FailureCategory.SANDBOX_FAILURE),
        ({"error_type": "RuntimeError", "origin": "orchestrator"}, FailureCategory.ORCHESTRATOR_FAILURE),
    ],
)
def test_failure_taxonomy_is_deterministic(kwargs, expected):
    assert classify_failure(**kwargs) is expected


@dataclass
class _Response:
    text: str = "ok"
    input_tokens: int = 11
    output_tokens: int = 7


class _Client:
    def __init__(self, error=None):
        self.error = error

    def complete(self, *args, **kwargs):
        if self.error:
            raise self.error
        return _Response()


class _Executor:
    def run(self, code, namespace=None):
        return "stdout", "", {"result": 1}


def test_offline_smoke_writes_atomic_manifest_and_append_only_ledgers(tmp_path):
    recorder = RunRecorder.create(tmp_path, _condition(), git_dirty=False)
    client = RecordingLLMClient(
        _Client(), recorder, provider="fixture", model="fixture-model"
    )
    executor = RecordingExecutor(_Executor(), recorder)

    response = client.complete(model="fixture-model", messages=[])
    assert response.text == "ok"
    assert executor.run("result = 1", {}) == ("stdout", "", {"result": 1})
    recorder.record_notebook_events(
        [{"action": "validate", "public": {"status": "ok"}, "private": {"score": 0.99}}]
    )
    recorder.finalize({"final_status": "submitted_clean", "valid_submit": True})

    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    assert manifest["state"] == "completed"
    assert manifest["failure_category"] == "success"
    assert manifest["ledger_totals"] == {
        "logical_llm_calls": 1,
        "provider_request_attempts": 0,
        "input_tokens": 11,
        "output_tokens": 7,
        "execution_events": 2,
    }
    usage = _read_jsonl(recorder.run_dir / "usage_ledger.jsonl")
    execution = _read_jsonl(recorder.run_dir / "execution_ledger.jsonl")
    assert [item["index"] for item in usage] == [0]
    assert [item["index"] for item in execution] == [0, 1]
    assert "private" not in json.dumps(execution)
    assert not list(recorder.run_dir.glob("*.tmp"))
    assert not list(recorder.run_dir.glob(".*.tmp"))


def test_sensitive_metadata_is_redacted(tmp_path):
    recorder = RunRecorder.create(
        tmp_path,
        _condition(),
        rerun_of="run_original",
        rerun_reason="Bearer abcdefghijklmnopqrstuvwxyz",
    )
    recorder.record_execution(
        {"event": "fixture", "api_key": "sk-super-secret-credential", "input_tokens": 4}
    )
    recorder.finalize(
        {
            "final_status": "no_candidate_found",
            "valid_submit": False,
            "password": "do-not-store",
        }
    )

    artifacts = "\n".join(path.read_text(encoding="utf-8") for path in recorder.run_dir.iterdir())
    assert "do-not-store" not in artifacts
    assert "super-secret-credential" not in artifacts
    assert "abcdefghijklmnopqrstuvwxyz" not in artifacts
    assert "input_tokens" in artifacts


def test_provider_exception_is_recorded_and_reraised(tmp_path):
    recorder = RunRecorder.create(tmp_path, _condition())
    client = RecordingLLMClient(
        _Client(RuntimeError("provider exploded")),
        recorder,
        provider="fixture",
        model="fixture-model",
    )
    with pytest.raises(RuntimeError, match="provider exploded"):
        client.complete()

    usage = _read_jsonl(recorder.run_dir / "usage_ledger.jsonl")
    assert usage[0]["success"] is False
    assert usage[0]["error_category"] == "provider_other"
    recorder.fail(RuntimeError("provider exploded"), origin="provider")
    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    assert manifest["failure_category"] == "provider_other"

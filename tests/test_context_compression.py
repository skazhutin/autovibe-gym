import copy
import hashlib
import json

import pytest

from gym.agent import GymAgent
from gym.context_compression import (
    CONTEXT_PACK_SCHEMA_VERSION,
    ContextCompressionError,
    ContextCompressionPolicy,
    compress_context,
)


def _feature_hash(features):
    payload = json.dumps(
        features,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hypothesis(index):
    return {
        "action": "validate",
        "model_var": f"model_{index}",
        "notebook_revision": index,
        "outcome": "validated",
        "validation_metric": 0.5 + index / 100,
        "feedback_keys": [],
        "candidate_id": f"candidate-{index}",
    }


def _incumbent(index):
    return {
        "candidate_id": f"candidate-{index}",
        "model_var": f"model_{index}",
        "notebook_revision": index,
        "metric_name": "accuracy",
        "metric_direction": "higher",
        "validation_metric": 0.5 + index / 100,
        "raw_inference_ready": True,
        "registration_index": index,
    }


def _state():
    features = [
        {"name": f"feature_{index}", "dtype": "float64"}
        for index in range(8)
    ]
    hypotheses = [_hypothesis(index) for index in range(5)]
    return {
        "schema_version": CONTEXT_PACK_SCHEMA_VERSION,
        "task_contract": {
            "protocol_version": "jupyter-v1",
            "target_column": "target",
            "metric_name": "accuracy",
            "metric_direction": "higher",
            "max_steps": 20,
            "train_rows": 100,
            "feedback_validation_rows": 20,
            "feature_schema": features,
            "feature_schema_sha256": _feature_hash(features),
        },
        "tested_hypotheses": hypotheses,
        "incumbent_state": _incumbent(2),
        "unresolved_errors": [
            {
                "action": "run_cell",
                "feedback_keys": [f"error_{index}"],
                "cell_id": f"cell_{index}",
            }
            for index in range(4)
        ],
        "remaining_resources": {
            "steps_remaining": 7,
            "step_limit": 20,
            "episode_budget": {
                "phase": "exploration",
                "global_remaining": {
                    "total_tokens": 1000,
                    "llm_calls": 4,
                    "code_executions": 6,
                    "tool_calls": 8,
                },
                "exploration_remaining": {
                    "total_tokens": 800,
                    "llm_calls": 3,
                    "code_executions": 4,
                    "tool_calls": 5,
                },
                "finalization_remaining": {
                    "total_tokens": 200,
                    "llm_calls": 1,
                    "code_executions": 2,
                    "tool_calls": 3,
                },
                "exploration_exhausted_reason": None,
                "validation_query_budget": None,
            },
        },
        "allowed_actions": ["add_cell", "validate", "finalize"],
        "notebook_state": {
            "dirty_since_clean_run": True,
            "clean_run_available": False,
            "validated_candidate_available": False,
            "notebook_revision": 9,
            "last_clean_run_id": None,
        },
        "finalization_contract": "Use only the host-owned incumbent.",
    }


def _policy(**overrides):
    values = {
        "max_context_bytes": 5000,
        "max_recent_messages": 3,
        "max_recent_message_bytes": 120,
        "max_feature_columns": 5,
        "max_tested_hypotheses": 3,
        "max_unresolved_errors": 2,
    }
    values.update(overrides)
    return ContextCompressionPolicy(**values)


def _decoded_pack(compressed):
    content = compressed.messages[0]["content"]
    return json.loads(content.split("\n", 1)[1])


def test_context_pack_is_byte_deterministic_and_mapping_order_independent():
    state = _state()
    reordered = {key: state[key] for key in reversed(list(state))}
    messages = [
        {"role": "user", "content": "initial task is represented structurally"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "second"},
    ]

    first = compress_context(state=state, messages=messages, policy=_policy())
    second = compress_context(state=reordered, messages=messages, policy=_policy())

    assert first.pack_bytes == second.pack_bytes
    assert first.messages == second.messages
    assert first.public_receipt() == second.public_receipt()
    assert first.context_bytes <= _policy().max_context_bytes
    assert first.pack_sha256 == hashlib.sha256(first.pack_bytes).hexdigest()
    assert first.public_receipt()["policy"] == _policy().to_dict()
    assert first.public_receipt()["source_context_bytes"] > 0


def test_required_fields_survive_deterministic_truncation_and_history_is_suffix():
    messages = [
        {"role": "user", "content": "PRIVATE_INITIAL_TASK_MUST_NOT_BE_COPIED"},
        {"role": "assistant", "content": "old-" + "x" * 300},
        {"role": "user", "content": "middle-" + "ы" * 300},
        {"role": "assistant", "content": "newest-" + "z" * 300},
    ]
    policy = _policy(
        max_context_bytes=2600,
        max_recent_messages=2,
        max_recent_message_bytes=48,
        max_feature_columns=2,
        max_tested_hypotheses=1,
        max_unresolved_errors=1,
    )

    compressed = compress_context(state=_state(), messages=messages, policy=policy)
    pack = _decoded_pack(compressed)
    text = json.dumps(compressed.as_messages(), ensure_ascii=False)

    assert compressed.context_bytes <= policy.max_context_bytes
    assert set(pack) == {
        "schema_version",
        "task_contract",
        "tested_hypotheses",
        "incumbent_state",
        "unresolved_errors",
        "remaining_resources",
        "allowed_actions",
        "notebook_state",
        "finalization_contract",
        "compression",
    }
    assert pack["incumbent_state"]["candidate_id"] == "candidate-2"
    assert pack["task_contract"]["feature_schema_sha256"] == _state()[
        "task_contract"
    ]["feature_schema_sha256"]
    assert len(pack["task_contract"]["feature_schema"]) <= 2
    assert len(pack["tested_hypotheses"]) <= 1
    assert len(pack["unresolved_errors"]) <= 1
    assert pack["remaining_resources"]["steps_remaining"] == 7
    assert pack["allowed_actions"] == ["add_cell", "validate", "finalize"]
    assert "PRIVATE_INITIAL_TASK_MUST_NOT_BE_COPIED" not in text
    assert "old-" not in text
    assert "newest-" in text
    assert compressed.truncated_recent_messages >= 1


def test_required_pack_fails_closed_instead_of_dropping_contract():
    with pytest.raises(ContextCompressionError, match="Required context pack"):
        compress_context(
            state=_state(),
            messages=[{"role": "user", "content": "initial"}],
            policy=_policy(max_context_bytes=100),
        )


def test_context_schema_rejects_private_or_unknown_fields_and_nonfinite_metrics():
    private_state = _state()
    private_state["private_diagnostics"] = {"secret": "value"}
    with pytest.raises(ContextCompressionError, match="top-level shape"):
        compress_context(
            state=private_state,
            messages=[{"role": "user", "content": "initial"}],
            policy=_policy(),
        )

    invalid_metric = _state()
    invalid_metric["tested_hypotheses"][0]["validation_metric"] = float("nan")
    with pytest.raises(ContextCompressionError, match="canonical JSON"):
        compress_context(
            state=invalid_metric,
            messages=[{"role": "user", "content": "initial"}],
            policy=_policy(),
        )


def test_context_pack_keeps_aggregate_validation_budget_and_rejects_source_ledger():
    state = _state()
    state["remaining_resources"]["episode_budget"]["validation_query_budget"] = {
        "policy_version": "validation-query-v1",
        "max_queries": 8,
        "finalization_reserve_queries": 1,
        "feedback_numeric_decimals": 3,
        "queries_used": 5,
        "phase_remaining_queries": 2,
        "global_remaining_queries": 3,
    }

    compressed = compress_context(
        state=state,
        messages=[{"role": "user", "content": "initial"}],
        policy=_policy(),
    )
    query_budget = _decoded_pack(compressed)["remaining_resources"][
        "episode_budget"
    ]["validation_query_budget"]

    assert query_budget["queries_used"] == 5
    assert query_budget["global_remaining_queries"] == 3
    assert "queries_by_source" not in query_budget

    private_state = copy.deepcopy(state)
    private_state["remaining_resources"]["episode_budget"][
        "validation_query_budget"
    ]["queries_by_source"] = {"private_tool": 5}
    with pytest.raises(ContextCompressionError, match="validation query budget"):
        compress_context(
            state=private_state,
            messages=[{"role": "user", "content": "initial"}],
            policy=_policy(),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_context_bytes", 0),
        ("max_recent_messages", -1),
        ("max_recent_message_bytes", 0),
        ("max_feature_columns", -1),
    ],
)
def test_context_policy_rejects_invalid_bounds(field, value):
    values = _policy().to_dict()
    values[field] = value
    with pytest.raises(ValueError):
        ContextCompressionPolicy(**values)


class _M3Environment:
    def __init__(self, state):
        self.state_payload = state
        self.receipts = []
        self.legacy_calls = 0

    def build_context_pack_v1(self):
        return copy.deepcopy(self.state_payload)

    def build_context_pack(self):
        self.legacy_calls += 1
        return {"legacy": True}

    def record_context_compression(self, receipt, *, private_receipt=None):
        self.receipts.append(
            {
                "public": dict(receipt),
                "private": dict(private_receipt or {}),
            }
        )


def test_agent_uses_explicit_m3_policy_and_records_content_free_receipt(monkeypatch):
    monkeypatch.setenv("AUTOVIBE_CONTEXT_COMPACTION", "conservative")
    env = _M3Environment(_state())
    agent = GymAgent(
        env=env,
        client=object(),
        context_compression_policy=_policy(),
    )
    agent.messages = [
        {"role": "user", "content": "legacy initial"},
        {"role": "assistant", "content": "recent assistant"},
    ]

    first = agent._messages_for_llm()
    second = agent._messages_for_llm()

    assert first == second
    assert env.receipts[0] == env.receipts[1]
    assert agent.last_context_compression_receipt == env.receipts[-1]["private"]
    receipt_text = json.dumps(env.receipts[-1]["public"])
    assert "recent assistant" not in receipt_text
    assert "legacy initial" not in receipt_text
    assert "context_sha256" not in env.receipts[-1]["public"]
    assert "source_context_sha256" not in env.receipts[-1]["public"]
    assert "context_sha256" in env.receipts[-1]["private"]
    assert env.legacy_calls == 0


def test_agent_default_path_remains_unmodified(monkeypatch):
    monkeypatch.delenv("AUTOVIBE_CONTEXT_COMPACTION", raising=False)
    env = _M3Environment(_state())
    agent = GymAgent(env=env, client=object())
    agent.messages = [{"role": "user", "content": "unchanged"}]

    assert agent._messages_for_llm() is agent.messages
    assert env.receipts == []


def test_agent_legacy_env_var_compressor_remains_compatible(monkeypatch):
    monkeypatch.setenv("AUTOVIBE_CONTEXT_COMPACTION", "conservative")
    monkeypatch.setenv("AUTOVIBE_CONTEXT_LAST_TURNS", "1")
    monkeypatch.setenv("AUTOVIBE_CONTEXT_MAX_CHARS", "5000")
    env = _M3Environment(_state())
    agent = GymAgent(env=env, client=object())
    agent.messages = [
        {"role": "user", "content": "initial"},
        {"role": "assistant", "content": "recent"},
    ]

    packed = agent._messages_for_llm()

    assert env.legacy_calls == 1
    assert packed[0] == agent.messages[0]
    assert "[CONTEXT PACK]" in packed[1]["content"]
    assert '"legacy": true' in packed[1]["content"]
    assert packed[-1] == agent.messages[-1]
    assert env.receipts == []

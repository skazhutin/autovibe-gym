"""Tests for the dashboard-driven prompt override path in GymAgent.

When the dashboard launcher sets ``AUTOVIBE_PROMPT_PAYLOAD_B64`` (or
``AUTOVIBE_PROMPT_PAYLOAD_FILE``), the agent must use the supplied blocks
and thoughts text instead of the canonical defaults. With no env var, the
behaviour is byte-identical to the historical default — this is the
critical invariant.
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace

from gym import agent as agent_module
from gym.prompts import build_system_prompt


# Lightweight stub instead of a real GymEnv: we only need attributes the
# agent reads during __init__ and the parts of run() we exercise.
class _FakeEnv:
    state = SimpleNamespace(max_steps=1)
    enable_thoughts = False

    def reset(self):
        return {"task": "noop"}


def _make_agent(monkeypatch, payload: dict | None):
    # Clear any vars from the parent process.
    monkeypatch.delenv("AUTOVIBE_PROMPT_PAYLOAD_B64", raising=False)
    monkeypatch.delenv("AUTOVIBE_PROMPT_PAYLOAD_FILE", raising=False)
    if payload is not None:
        raw = json.dumps(payload).encode("utf-8")
        b64 = base64.b64encode(raw).decode("ascii")
        monkeypatch.setenv("AUTOVIBE_PROMPT_PAYLOAD_B64", b64)
    # Construct without calling __init__ to avoid touching network/LLM clients.
    agent = agent_module.GymAgent.__new__(agent_module.GymAgent)
    agent.env = _FakeEnv()
    agent.model = "fake"
    agent.max_tokens = 8
    agent.client = None
    agent.messages = []
    agent.total_input_tokens = 0
    agent.total_output_tokens = 0
    agent._prompt_overrides = agent_module._load_prompt_overrides()
    agent.prompt_preset_id = str(
        agent._prompt_overrides.get("preset_id", "default") or "default"
    )
    raw_sha = agent._prompt_overrides.get("sha256")
    agent.prompt_sha256 = raw_sha if isinstance(raw_sha, str) else None
    return agent


def test_no_env_uses_default_preset(monkeypatch):
    agent = _make_agent(monkeypatch, None)
    assert agent.prompt_preset_id == "default"
    assert agent.prompt_sha256 is None
    # Assembled prompt matches the canonical default.
    overrides = agent._prompt_overrides
    blocks_override = overrides.get("blocks") if isinstance(overrides.get("blocks"), dict) else None
    out = build_system_prompt(blocks_override, thoughts_on=False)
    assert out == build_system_prompt(thoughts_on=False)


def test_b64_payload_overrides_block(monkeypatch):
    payload = {
        "preset_id": "minimal",
        "blocks": {"failure_patterns": "AVOID: nothing in particular."},
        "thoughts_on_text": None,
        "thoughts_off_text": None,
        "sha256": "deadbeef",
    }
    agent = _make_agent(monkeypatch, payload)
    assert agent.prompt_preset_id == "minimal"
    assert agent.prompt_sha256 == "deadbeef"
    out = build_system_prompt(
        agent._prompt_overrides["blocks"], thoughts_on=False,
    )
    assert "AVOID: nothing in particular." in out
    # And differs from the canonical default.
    assert out != build_system_prompt(thoughts_on=False)


def test_thoughts_text_override_applied(monkeypatch):
    payload = {
        "preset_id": "talky",
        "blocks": {},
        "thoughts_on_text": "\n\nCUSTOM ON.\n",
        "thoughts_off_text": None,
        "sha256": "x",
    }
    agent = _make_agent(monkeypatch, payload)
    overrides = agent._prompt_overrides
    out_on = build_system_prompt(
        overrides.get("blocks") or {},
        thoughts_on=True,
        thoughts_on_text=overrides.get("thoughts_on_text"),
    )
    assert out_on.endswith("\n\nCUSTOM ON.\n")


def test_payload_file_path_is_read(monkeypatch, tmp_path):
    payload = {
        "preset_id": "from-file",
        "blocks": {"finalize": "FINALIZE FAST."},
        "thoughts_on_text": None,
        "thoughts_off_text": None,
        "sha256": "abc123",
    }
    path = tmp_path / "prompt_payload.json"
    path.write_text(json.dumps(payload), "utf-8")
    monkeypatch.delenv("AUTOVIBE_PROMPT_PAYLOAD_B64", raising=False)
    monkeypatch.setenv("AUTOVIBE_PROMPT_PAYLOAD_FILE", str(path))
    loaded = agent_module._load_prompt_overrides()
    assert loaded["preset_id"] == "from-file"
    assert loaded["blocks"]["finalize"] == "FINALIZE FAST."


def test_b64_takes_precedence_over_file(monkeypatch, tmp_path):
    """If both env vars are set, the inline b64 wins (no disk read needed)."""
    path = tmp_path / "ignored.json"
    path.write_text(json.dumps({"preset_id": "file-wins"}), "utf-8")
    b64_payload = {"preset_id": "b64-wins"}
    monkeypatch.setenv(
        "AUTOVIBE_PROMPT_PAYLOAD_B64",
        base64.b64encode(json.dumps(b64_payload).encode()).decode("ascii"),
    )
    monkeypatch.setenv("AUTOVIBE_PROMPT_PAYLOAD_FILE", str(path))
    assert agent_module._load_prompt_overrides()["preset_id"] == "b64-wins"


def test_corrupt_b64_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("AUTOVIBE_PROMPT_PAYLOAD_B64", "@@@not-base64@@@")
    monkeypatch.delenv("AUTOVIBE_PROMPT_PAYLOAD_FILE", raising=False)
    assert agent_module._load_prompt_overrides() == {}


def test_corrupt_json_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(
        "AUTOVIBE_PROMPT_PAYLOAD_B64",
        base64.b64encode(b"{not json").decode("ascii"),
    )
    assert agent_module._load_prompt_overrides() == {}


def test_missing_file_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOVIBE_PROMPT_PAYLOAD_B64", raising=False)
    monkeypatch.setenv("AUTOVIBE_PROMPT_PAYLOAD_FILE", str(tmp_path / "missing.json"))
    assert agent_module._load_prompt_overrides() == {}

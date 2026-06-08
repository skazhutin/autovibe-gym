"""Tests for the system-prompt preset path inside run_launcher.

We exercise the small pure-function helpers (_resolve_prompt_payload,
_build_env) without spawning a subprocess. The full launch() path needs
real binaries so we keep it for the smoke test in dashboard up-tests.
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from dashboard.server.app.services import prompt_store, run_launcher


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        prompts_dir=tmp_path / "prompts",
        repo_root=tmp_path,
        mlflow_tracking_uri="sqlite:///" + str(tmp_path / "mlflow.db"),
    )
    settings.prompts_dir.mkdir()
    settings.ensure_dirs = lambda: settings.prompts_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(prompt_store, "get_settings", lambda: settings)
    # _build_env also calls get_settings; redirect that too.
    monkeypatch.setattr(run_launcher, "get_settings", lambda: settings)
    # _llm_env reads model_store; stub it for cfg without modelId so we can
    # exercise env assembly without the model registry.
    monkeypatch.setattr(run_launcher.model_store, "get_model", lambda _id: None)
    return settings


def _decode(env: dict) -> dict | None:
    raw = env.get("AUTOVIBE_PROMPT_PAYLOAD_B64")
    if not raw:
        return None
    return json.loads(base64.b64decode(raw).decode("utf-8"))


def test_resolve_payload_default_for_gym(isolated_store):
    cfg = {"mode": "gym", "enableThoughts": True}
    payload = run_launcher._resolve_prompt_payload(cfg)
    assert payload is not None
    assert payload["preset_id"] == "default"
    assert "Thoughts mode is enabled." in payload["assembled_prompt"]


def test_resolve_payload_uses_selected_preset(isolated_store):
    prompt_store.save_preset({
        "id": "minimal", "name": "Minimal",
        "blocks": {"failure_patterns": "AVOID: nothing in particular."},
        "thoughts_on": None, "thoughts_off": None,
    })
    cfg = {"mode": "iterative", "promptPresetId": "minimal", "enableThoughts": False}
    payload = run_launcher._resolve_prompt_payload(cfg)
    assert payload is not None
    assert payload["preset_id"] == "minimal"
    assert "AVOID: nothing in particular." in payload["assembled_prompt"]


def test_resolve_payload_falls_back_when_preset_missing(isolated_store):
    """A preset deleted between selection and launch must not brick the run."""
    cfg = {"mode": "gym", "promptPresetId": "does-not-exist"}
    payload = run_launcher._resolve_prompt_payload(cfg)
    assert payload is not None
    assert payload["preset_id"] == "default"


def test_resolve_payload_skipped_for_single_shot(isolated_store):
    cfg = {"mode": "single"}
    assert run_launcher._resolve_prompt_payload(cfg) is None


def test_resolve_payload_skipped_for_repeated(isolated_store):
    cfg = {"mode": "repeated"}
    assert run_launcher._resolve_prompt_payload(cfg) is None


def test_build_env_injects_b64_payload(isolated_store):
    payload = run_launcher._resolve_prompt_payload(
        {"mode": "gym", "enableThoughts": False}
    )
    cfg = {"mode": "gym", "modelId": None, "_promptPayload": payload}
    env = run_launcher._build_env(cfg)
    decoded = _decode(env)
    assert decoded is not None
    assert decoded["preset_id"] == "default"
    # Only the slim payload survives — assembled_prompt is reconstructed by the agent.
    assert "assembled_prompt" not in decoded
    assert decoded["sha256"] == payload["sha256"]


def test_build_env_skips_payload_when_absent(isolated_store):
    """Single-shot / repeated runs never set _promptPayload — env stays clean."""
    cfg = {"mode": "single", "modelId": None}
    env = run_launcher._build_env(cfg)
    assert "AUTOVIBE_PROMPT_PAYLOAD_B64" not in env

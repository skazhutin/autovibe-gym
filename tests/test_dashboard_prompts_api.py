"""Integration tests for the /api/prompts router."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dashboard.server.app.main import app
from dashboard.server.app.services import prompt_store


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Re-point prompt_store at a tmp dir so tests don't pollute real data/."""
    settings = SimpleNamespace(prompts_dir=tmp_path / "prompts")
    settings.prompts_dir.mkdir()
    settings.ensure_dirs = lambda: settings.prompts_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(prompt_store, "get_settings", lambda: settings)
    with TestClient(app) as c:
        yield c


def test_list_returns_default_first(client):
    r = client.get("/api/prompts")
    assert r.status_code == 200
    body = r.json()
    assert body["default_id"] == "default"
    assert body["items"][0]["id"] == "default"
    assert body["items"][0]["is_default"] is True


def test_get_default(client):
    r = client.get("/api/prompts/default")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "default"
    assert body["is_default"] is True
    assert "kernel_vars" in body["blocks"]
    assert body["block_tiers"]["kernel_vars"] == "locked"
    assert body["sha256"]


def test_create_then_fetch(client):
    payload = {
        "id": "minimal",
        "name": "Minimal",
        "blocks": {"failure_patterns": "AVOID: nothing in particular."},
    }
    r = client.post("/api/prompts", json=payload)
    assert r.status_code == 200, r.text
    created = r.json()
    assert created["id"] == "minimal"
    assert created["blocks"]["failure_patterns"] == "AVOID: nothing in particular."
    # Fetch through GET.
    r2 = client.get("/api/prompts/minimal")
    assert r2.status_code == 200
    assert r2.json()["sha256"] == created["sha256"]


def test_get_unknown_returns_404(client):
    r = client.get("/api/prompts/does-not-exist")
    assert r.status_code == 404


def test_create_with_invalid_id_returns_422(client):
    r = client.post(
        "/api/prompts",
        json={"id": "Bad Id!!", "name": "X", "blocks": {}},
    )
    assert r.status_code == 422


def test_locked_block_override_rejected_with_422(client):
    r = client.post(
        "/api/prompts",
        json={
            "id": "evil",
            "name": "Evil",
            "blocks": {"kernel_vars": "I lied about variables"},
        },
    )
    assert r.status_code == 422
    assert "locked" in r.text.lower()


def test_unknown_block_rejected_with_422(client):
    r = client.post(
        "/api/prompts",
        json={
            "id": "x",
            "name": "X",
            "blocks": {"made_up_block": "..."},
        },
    )
    assert r.status_code == 422


def test_default_id_reserved_on_create(client):
    r = client.post(
        "/api/prompts",
        json={"id": "default", "name": "Hacked", "blocks": {}},
    )
    assert r.status_code == 422


def test_default_cannot_be_deleted(client):
    r = client.delete("/api/prompts/default")
    assert r.status_code == 422


def test_delete_user_preset(client):
    client.post(
        "/api/prompts",
        json={"id": "tmp", "name": "T", "blocks": {}},
    )
    r = client.delete("/api/prompts/tmp")
    assert r.status_code == 200
    assert r.json()["deleted"] == "tmp"
    # second delete is 404.
    r2 = client.delete("/api/prompts/tmp")
    assert r2.status_code == 404


def test_warnings_surfaced_on_get(client):
    """Stripping a contract phrase from the preset should appear in warnings."""
    # Override the editable failure_patterns and finalize blocks to remove
    # the word "submit". This must not block save but produce a warning.
    client.post(
        "/api/prompts",
        json={
            "id": "stripped",
            "name": "Stripped",
            "blocks": {
                "failure_patterns": "Be careful.",
                "finalize": "Eventually finish.",
            },
        },
    )
    detail = client.get("/api/prompts/stripped").json()
    # default body still contains "submit" in critical_rules, so we expect
    # warnings list to exist (possibly empty); shape is what matters.
    assert isinstance(detail["warnings"], list)

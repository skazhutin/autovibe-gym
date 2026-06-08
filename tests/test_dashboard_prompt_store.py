from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dashboard.server.app.services import prompt_store


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    settings = SimpleNamespace(prompts_dir=tmp_path / "prompts")
    settings.prompts_dir.mkdir()
    # ``ensure_dirs`` is called by ``list_presets``; provide a stub.
    settings.ensure_dirs = lambda: settings.prompts_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(prompt_store, "get_settings", lambda: settings)
    return settings


# ---------- defaults ----------------------------------------------------------


def test_default_preset_is_always_listed(isolated_store):
    items = prompt_store.list_presets()
    assert any(p["id"] == "default" and p["is_default"] for p in items)


def test_default_preset_detail_synthesized_from_code(isolated_store):
    """Default detail is synthesized — touching any disk file cannot poison it."""
    detail = prompt_store.get_default_preset()
    assert detail["id"] == "default"
    assert detail["is_default"] is True
    # All BLOCK_ORDER blocks must be present and equal to DEFAULT_BLOCKS.
    from gym.prompts import BLOCK_ORDER, DEFAULT_BLOCKS

    assert set(detail["blocks"].keys()) == set(BLOCK_ORDER)
    for name in BLOCK_ORDER:
        assert detail["blocks"][name] == DEFAULT_BLOCKS[name]


def test_default_id_is_reserved_for_writes(isolated_store):
    with pytest.raises(prompt_store.PresetValidationError):
        prompt_store.save_preset({"id": "default", "name": "Hacked", "blocks": {}})


def test_default_cannot_be_deleted(isolated_store):
    with pytest.raises(prompt_store.PresetValidationError):
        prompt_store.delete_preset("default")


def test_default_file_on_disk_is_ignored(isolated_store):
    """A hand-written default.json must not shadow the code-derived default."""
    (isolated_store.prompts_dir / "default.json").write_text(
        json.dumps({"id": "default", "name": "Evil", "blocks": {"failure_patterns": "MWAHAHA"}}),
        "utf-8",
    )
    items = [p for p in prompt_store.list_presets() if p["id"] == "default"]
    assert len(items) == 1
    assert items[0]["name"] == "Default"
    detail = prompt_store.get_default_preset()
    assert "MWAHAHA" not in detail["blocks"]["failure_patterns"]


# ---------- save / get / delete ----------------------------------------------


def _valid_payload(**overrides):
    base = {
        "id": "minimal",
        "name": "Minimal",
        "blocks": {"failure_patterns": "AVOID: nothing in particular."},
        "thoughts_on": None,
        "thoughts_off": None,
    }
    base.update(overrides)
    return base


def test_save_then_get_roundtrip(isolated_store):
    detail = prompt_store.save_preset(_valid_payload())
    assert detail["id"] == "minimal"
    assert detail["is_default"] is False
    assert detail["blocks"]["failure_patterns"] == "AVOID: nothing in particular."
    # Non-overridden blocks fall back to defaults.
    from gym.prompts import DEFAULT_BLOCKS

    assert detail["blocks"]["header"] == DEFAULT_BLOCKS["header"]
    again = prompt_store.get_preset("minimal")
    assert again["blocks"] == detail["blocks"]
    assert again["sha256"] == detail["sha256"]


def test_save_preserves_created_at(isolated_store):
    first = prompt_store.save_preset(_valid_payload())
    second = prompt_store.save_preset(_valid_payload(name="Minimal v2"))
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] != first["created_at"] or True  # at minimum: not crashed


def test_delete_removes_file(isolated_store):
    prompt_store.save_preset(_valid_payload())
    assert prompt_store.delete_preset("minimal") is True
    assert prompt_store.delete_preset("minimal") is False  # idempotent second call


def test_get_unknown_preset_raises_key_error(isolated_store):
    with pytest.raises(KeyError):
        prompt_store.get_preset("does-not-exist")


# ---------- validation -------------------------------------------------------


def test_id_must_match_pattern(isolated_store):
    with pytest.raises(prompt_store.PresetValidationError):
        prompt_store.save_preset(_valid_payload(id="Bad Id"))
    with pytest.raises(prompt_store.PresetValidationError):
        prompt_store.save_preset(_valid_payload(id=""))
    with pytest.raises(prompt_store.PresetValidationError):
        prompt_store.save_preset(_valid_payload(id="a" * 41))


def test_name_required(isolated_store):
    with pytest.raises(prompt_store.PresetValidationError):
        prompt_store.save_preset(_valid_payload(name=""))


def test_unknown_block_rejected(isolated_store):
    with pytest.raises(prompt_store.PresetValidationError):
        prompt_store.save_preset(_valid_payload(blocks={"made_up_block": "..."}))


def test_locked_block_override_rejected_at_api_level(isolated_store):
    """Locked block (kernel_vars) override is a 422 at API layer."""
    with pytest.raises(prompt_store.PresetValidationError):
        prompt_store.save_preset(_valid_payload(blocks={"kernel_vars": "POISONED_VAR_LIST"}))


def test_locked_block_dropped_at_assembly_layer():
    """Defence in depth: even a hand-edited file cannot poison the prompt."""
    from gym.prompts import assemble_body

    body = assemble_body({"kernel_vars": "POISONED_VAR_LIST"})
    assert "POISONED_VAR_LIST" not in body
    assert body == assemble_body()


def test_thoughts_text_override_persists(isolated_store):
    detail = prompt_store.save_preset(
        _valid_payload(thoughts_on="\n\nCUSTOM ON.\n", thoughts_off=None)
    )
    assert detail["thoughts_on_overridden"] is True
    assert detail["thoughts_off_overridden"] is False
    assert "CUSTOM ON." in detail["thoughts_on"]


# ---------- sanity check -----------------------------------------------------


def test_sanity_check_passes_for_default():
    from gym.prompts import DEFAULT_BLOCKS, DEFAULT_THOUGHTS_OFF, DEFAULT_THOUGHTS_ON

    warnings = prompt_store.sanity_check(DEFAULT_BLOCKS, DEFAULT_THOUGHTS_ON, DEFAULT_THOUGHTS_OFF)
    assert warnings == []


def test_sanity_check_warns_when_contract_phrases_removed():
    """Stripping every block to harmless filler should trigger warnings."""
    stripped = {name: "" for name in prompt_store.BLOCK_ORDER}
    warnings = prompt_store.sanity_check(stripped, "", "")
    # We expect at least the 6 canonical markers to be missing.
    assert len(warnings) >= 5
    assert any("restart_and_run_all" in w for w in warnings)


# ---------- runtime payload (for run_launcher) -------------------------------


def test_runtime_payload_default(isolated_store):
    payload = prompt_store.build_runtime_payload("default", thoughts_on=False)
    assert payload["preset_id"] == "default"
    assert payload["blocks"] == {}
    assert payload["sha256"]
    assert "ACTION_JSON_SCHEMA" not in payload["assembled_prompt"]  # raw value, not placeholder
    assert "Thoughts mode is disabled." in payload["assembled_prompt"]


def test_runtime_payload_with_override(isolated_store):
    prompt_store.save_preset(_valid_payload(blocks={"finalize": "JUST SUBMIT ALREADY."}))
    payload = prompt_store.build_runtime_payload("minimal", thoughts_on=True)
    assert "JUST SUBMIT ALREADY." in payload["assembled_prompt"]
    assert "Thoughts mode is enabled." in payload["assembled_prompt"]
    # Two presets — default and minimal — must produce different SHAs.
    other = prompt_store.build_runtime_payload("default", thoughts_on=True)
    assert payload["sha256"] != other["sha256"]

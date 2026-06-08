"""System-prompt preset store.

Storage layout: one JSON file per preset at ``<data>/prompts/<id>.json``.
The ``default`` preset is NEVER stored on disk — it is synthesized at read
time from ``gym.prompts.DEFAULT_BLOCKS`` so that "Reset to default" always
returns the code's current source of truth, even if a previous default file
was hand-edited.

On-disk preset schema (only override fields are stored; missing blocks fall
back to ``DEFAULT_BLOCKS`` when assembled)::

    {
      "id": "minimal",
      "name": "Minimal",
      "blocks": {"failure_patterns": "AVOID: ..."},
      "thoughts_on": null,   // null → use DEFAULT_THOUGHTS_ON
      "thoughts_off": null,
      "created_at": "2026-06-08T14:00:00Z",
      "updated_at": "2026-06-08T14:00:00Z"
    }

Validation policy:
- ``id`` must match ``[a-z0-9][a-z0-9_-]*``, length 1..40.
- ``id == "default"`` is reserved (cannot be created, updated, or deleted).
- Block keys outside ``BLOCK_ORDER`` are rejected.
- Locked-block overrides are rejected (kernel_vars and any future locked
  blocks tie to runtime contract; the dashboard layer must surface this as
  a clear 422 — silent dropping in ``gym.prompts.assemble_body`` is only a
  defence-in-depth.

Sanity-check: produces *warnings* (not errors) when the assembled body
loses canonical contract phrases (``restart_and_run_all``, ``validate``,
``submit``, ``model_var``, ``predict``, ``raw``). The dashboard surfaces
these so an operator notices that they have removed important guidance.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from ..config import get_settings
from gym.prompts import (
    BLOCK_ORDER,
    BLOCK_TIERS,
    DEFAULT_BLOCKS,
    DEFAULT_THOUGHTS_OFF,
    DEFAULT_THOUGHTS_ON,
    LOCKED_BLOCKS,
    assemble_body,
    build_system_prompt,
)


DEFAULT_PRESET_ID = "default"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_RESERVED_IDS = frozenset({DEFAULT_PRESET_ID})

# Phrases the dashboard warns about when removed from the assembled body.
# Each entry is (regex, human-readable explanation).
_CONTRACT_MARKERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"restart_and_run_all", re.IGNORECASE),
     "Mention of `restart_and_run_all` is missing — the agent may skip the clean reproducibility step."),
    (re.compile(r"\bvalidate\b", re.IGNORECASE),
     "Mention of `validate` is missing — the agent may submit without prior validation."),
    (re.compile(r"\bsubmit\b", re.IGNORECASE),
     "Mention of `submit` is missing — the agent may not realise it must submit."),
    (re.compile(r"model_var", re.IGNORECASE),
     "Mention of `model_var` is missing — the agent may not name the candidate variable correctly."),
    (re.compile(r"\bpredict\b", re.IGNORECASE),
     "Mention of `predict` is missing — the raw-rows inference contract is not stated."),
    (re.compile(r"\braw\b", re.IGNORECASE),
     "Mention of `raw` rows/data is missing — the agent may submit a model that breaks on raw input."),
]


class PresetValidationError(ValueError):
    """Raised when a preset payload is structurally invalid."""


# ---------- file-system primitives -------------------------------------------


def _preset_path(preset_id: str) -> Any:
    return get_settings().prompts_dir / f"{preset_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_file(preset_id: str) -> dict[str, Any]:
    path = _preset_path(preset_id)
    if not path.exists():
        raise FileNotFoundError(preset_id)
    return json.loads(path.read_text("utf-8"))


def _write_file(preset_id: str, data: dict[str, Any]) -> None:
    path = _preset_path(preset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic-ish: write tmp + rename. On Windows os.replace is atomic.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
    tmp.replace(path)


# ---------- preset assembly --------------------------------------------------


def _default_preset_record() -> dict[str, Any]:
    """Synthetic record for the always-available `default` preset."""
    return {
        "id": DEFAULT_PRESET_ID,
        "name": "Default",
        "blocks": {},
        "thoughts_on": None,
        "thoughts_off": None,
        "created_at": None,
        "updated_at": None,
        "is_default": True,
    }


def _resolve_full_blocks(stored_blocks: dict[str, str]) -> dict[str, str]:
    """Merge stored overrides with DEFAULT_BLOCKS to produce a complete map."""
    full = dict(DEFAULT_BLOCKS)
    for name, value in (stored_blocks or {}).items():
        if name in BLOCK_ORDER and name not in LOCKED_BLOCKS:
            full[name] = value
    return full


def _record_to_detail(record: dict[str, Any]) -> dict[str, Any]:
    """Expand a stored record into a detail dict with full blocks + assembled prompt."""
    stored_blocks = record.get("blocks") or {}
    full_blocks = _resolve_full_blocks(stored_blocks)
    thoughts_on_text = record.get("thoughts_on") or DEFAULT_THOUGHTS_ON
    thoughts_off_text = record.get("thoughts_off") or DEFAULT_THOUGHTS_OFF
    # SHA covers the user's effective config (overrides + thoughts toggles).
    # We hash assembled-body + both thoughts variants so any change shifts it.
    body = assemble_body(stored_blocks)
    sha = hashlib.sha256(
        (body + "\n--ON--\n" + thoughts_on_text + "\n--OFF--\n" + thoughts_off_text).encode("utf-8")
    ).hexdigest()
    return {
        "id": record["id"],
        "name": record["name"],
        "blocks": full_blocks,
        "block_overrides": {k: v for k, v in (stored_blocks or {}).items() if k in BLOCK_ORDER},
        "thoughts_on": thoughts_on_text,
        "thoughts_off": thoughts_off_text,
        "thoughts_on_overridden": record.get("thoughts_on") is not None,
        "thoughts_off_overridden": record.get("thoughts_off") is not None,
        "block_tiers": dict(BLOCK_TIERS),
        "block_order": list(BLOCK_ORDER),
        "locked_blocks": sorted(LOCKED_BLOCKS),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "is_default": bool(record.get("is_default", False)),
        "sha256": sha,
        "warnings": sanity_check(full_blocks, thoughts_on_text, thoughts_off_text),
    }


# ---------- validation -------------------------------------------------------


def _validate_id(preset_id: str, *, allow_reserved: bool = False) -> None:
    if not isinstance(preset_id, str):
        raise PresetValidationError("id must be a string")
    if not _ID_RE.match(preset_id):
        raise PresetValidationError(
            "id must match [a-z0-9][a-z0-9_-]* and be 1..40 chars long"
        )
    if not allow_reserved and preset_id in _RESERVED_IDS:
        raise PresetValidationError(f"id {preset_id!r} is reserved")


def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise & validate user-supplied preset payload (for create/update)."""
    if not isinstance(payload, dict):
        raise PresetValidationError("preset payload must be an object")
    preset_id = payload.get("id")
    name = payload.get("name")
    blocks = payload.get("blocks") or {}
    thoughts_on = payload.get("thoughts_on")
    thoughts_off = payload.get("thoughts_off")

    _validate_id(str(preset_id) if preset_id is not None else "")

    if not isinstance(name, str) or not name.strip():
        raise PresetValidationError("name is required")
    if len(name) > 80:
        raise PresetValidationError("name must be 80 chars or shorter")

    if not isinstance(blocks, dict):
        raise PresetValidationError("blocks must be an object")
    cleaned_blocks: dict[str, str] = {}
    for k, v in blocks.items():
        if k not in BLOCK_ORDER:
            raise PresetValidationError(f"unknown block name: {k!r}")
        if k in LOCKED_BLOCKS:
            raise PresetValidationError(
                f"block {k!r} is locked and cannot be overridden"
            )
        if not isinstance(v, str):
            raise PresetValidationError(f"block {k!r} must be a string")
        # Empty string is allowed (clears the block) but we treat null as "use default".
        cleaned_blocks[k] = v

    for field_name, value in (("thoughts_on", thoughts_on), ("thoughts_off", thoughts_off)):
        if value is not None and not isinstance(value, str):
            raise PresetValidationError(f"{field_name} must be a string or null")

    return {
        "id": preset_id,
        "name": name.strip(),
        "blocks": cleaned_blocks,
        "thoughts_on": thoughts_on,
        "thoughts_off": thoughts_off,
    }


def sanity_check(
    full_blocks: dict[str, str],
    thoughts_on_text: str,
    thoughts_off_text: str,
) -> list[str]:
    """Emit non-blocking warnings when canonical contract markers are missing.

    We assemble a representative text that the agent would actually see
    (body + both thoughts variants) and grep for canonical phrases. If the
    operator removed every mention of, say, ``restart_and_run_all``, the
    dashboard surfaces a warning but still allows the save.
    """
    body = "\n\n".join(full_blocks[name] for name in BLOCK_ORDER if name in full_blocks)
    combined = body + "\n" + thoughts_on_text + "\n" + thoughts_off_text
    warnings: list[str] = []
    for regex, message in _CONTRACT_MARKERS:
        if not regex.search(combined):
            warnings.append(message)
    return warnings


# ---------- public API -------------------------------------------------------


def list_presets() -> list[dict[str, Any]]:
    """Return all presets including the synthetic ``default``.

    The summary shape is intentionally compact (id, name, is_default,
    block_override_count, updated_at) so the /prompts page can render the
    sidebar without parsing every full body.
    """
    items: list[dict[str, Any]] = []
    settings = get_settings()
    settings.ensure_dirs()
    items.append({
        "id": DEFAULT_PRESET_ID,
        "name": "Default",
        "is_default": True,
        "block_override_count": 0,
        "updated_at": None,
    })
    for path in sorted(settings.prompts_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        pid = data.get("id") or path.stem
        if pid == DEFAULT_PRESET_ID:
            # User-written `default.json` is ignored — code is the source of truth.
            continue
        items.append({
            "id": pid,
            "name": data.get("name") or pid,
            "is_default": False,
            "block_override_count": len(data.get("blocks") or {}),
            "updated_at": data.get("updated_at"),
        })
    return items


def get_preset(preset_id: str) -> dict[str, Any]:
    """Return a fully-resolved preset detail (raises KeyError if missing)."""
    if preset_id == DEFAULT_PRESET_ID:
        return _record_to_detail(_default_preset_record())
    try:
        record = _read_file(preset_id)
    except FileNotFoundError as exc:
        raise KeyError(preset_id) from exc
    record.setdefault("id", preset_id)
    record.setdefault("name", preset_id)
    record["is_default"] = False
    return _record_to_detail(record)


def get_default_preset() -> dict[str, Any]:
    return _record_to_detail(_default_preset_record())


def save_preset(payload: dict[str, Any]) -> dict[str, Any]:
    """Create or update a preset. Raises PresetValidationError on bad input."""
    cleaned = _validate_payload(payload)
    preset_id = cleaned["id"]
    now = _now()
    # Preserve created_at across updates.
    try:
        existing = _read_file(preset_id)
        created_at = existing.get("created_at") or now
    except FileNotFoundError:
        created_at = now
    record = {
        "id": preset_id,
        "name": cleaned["name"],
        "blocks": cleaned["blocks"],
        "thoughts_on": cleaned["thoughts_on"],
        "thoughts_off": cleaned["thoughts_off"],
        "created_at": created_at,
        "updated_at": now,
    }
    _write_file(preset_id, record)
    return _record_to_detail({**record, "is_default": False})


def delete_preset(preset_id: str) -> bool:
    """Delete a user preset. Returns True if a file was removed."""
    if preset_id == DEFAULT_PRESET_ID:
        raise PresetValidationError("default preset cannot be deleted")
    _validate_id(preset_id)
    path = _preset_path(preset_id)
    if not path.exists():
        return False
    path.unlink()
    return True


# ---------- runtime helpers (for run_launcher) -------------------------------


def build_runtime_payload(preset_id: str, *, thoughts_on: bool) -> dict[str, Any]:
    """Return a dict the agent process can consume to reconstruct the prompt.

    Shape::

        {
          "preset_id": "minimal",
          "blocks": {...overrides only, no defaults...},
          "thoughts_on_text": "..." | null,
          "thoughts_off_text": "..." | null,
          "assembled_prompt": "<full text the agent will see>",
          "sha256": "<sha of the assembled prompt>"
        }

    The launcher dumps this to a temp JSON file and the agent loads it on
    init (next task). ``assembled_prompt`` is precomputed for MLflow logging
    and so the agent doesn't have to re-run ``build_system_prompt``.
    """
    if preset_id == DEFAULT_PRESET_ID:
        record = _default_preset_record()
    else:
        try:
            record = _read_file(preset_id)
        except FileNotFoundError as exc:
            raise KeyError(preset_id) from exc
    stored_blocks = record.get("blocks") or {}
    thoughts_on_text = record.get("thoughts_on")
    thoughts_off_text = record.get("thoughts_off")
    assembled = build_system_prompt(
        stored_blocks,
        thoughts_on=thoughts_on,
        thoughts_on_text=thoughts_on_text,
        thoughts_off_text=thoughts_off_text,
    )
    return {
        "preset_id": preset_id,
        "blocks": dict(stored_blocks),
        "thoughts_on_text": thoughts_on_text,
        "thoughts_off_text": thoughts_off_text,
        "assembled_prompt": assembled,
        "sha256": hashlib.sha256(assembled.encode("utf-8")).hexdigest(),
    }

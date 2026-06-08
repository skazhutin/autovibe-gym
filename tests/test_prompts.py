"""Regression and behaviour tests for gym.prompts.

The byte-identity tests pin the assembled default prompt to a SHA256 hash.
The hashes were captured from the legacy monolithic ``SYSTEM_PROMPT`` /
``THOUGHTS_*_PROMPT`` constants before the block-structured refactor. If a
later change to DEFAULT_BLOCKS or DEFAULT_THOUGHTS_* breaks these tests, that
is a deliberate baseline shift: it means existing experiment results are no
longer directly comparable to runs from before the change. Update the SHA
here intentionally, document why in docs/STATUS.md, and inform the team.
"""
from __future__ import annotations

import hashlib

import pytest

from gym import agent as agent_module
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


# Captured from the pre-refactor SYSTEM_PROMPT + THOUGHTS_*_PROMPT constants.
# Do not change without a deliberate baseline shift.
SHA_BODY_DEFAULT = "c06f4bac365968665acb5da73a42ff40d6910572d253344c56e10985d36df991"
SHA_THOUGHTS_ON = "3813dd05d7bad29f565c6de52f10d14b6f7306933999314aed537dfebaba6d24"
SHA_THOUGHTS_OFF = "c8bf0bdac0a02aebaf1366fe82585d5327f0987f3fe3272da0771b069da92a1f"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------- baseline identity -------------------------------------------------


def test_default_body_sha_is_pinned():
    """Default body (header → JSON schema) must match historical SHA."""
    assert _sha(assemble_body()) == SHA_BODY_DEFAULT


def test_thoughts_on_sha_is_pinned():
    """Default prompt with thoughts ON must match historical SHA."""
    assert _sha(build_system_prompt(thoughts_on=True)) == SHA_THOUGHTS_ON


def test_thoughts_off_sha_is_pinned():
    """Default prompt with thoughts OFF must match historical SHA."""
    assert _sha(build_system_prompt(thoughts_on=False)) == SHA_THOUGHTS_OFF


def test_agent_reexports_match_canonical():
    """gym.agent re-exports must equal gym.prompts canonical values."""
    assert agent_module.SYSTEM_PROMPT == assemble_body()
    assert agent_module.THOUGHTS_ENABLED_PROMPT == DEFAULT_THOUGHTS_ON
    assert agent_module.THOUGHTS_DISABLED_PROMPT == DEFAULT_THOUGHTS_OFF


# ---------- structural invariants --------------------------------------------


def test_block_order_keys_match_default_blocks():
    """BLOCK_ORDER and DEFAULT_BLOCKS must agree on the same set of names."""
    assert set(BLOCK_ORDER) == set(DEFAULT_BLOCKS.keys())


def test_block_tiers_cover_all_blocks():
    """Every block in BLOCK_ORDER must have an explicit tier."""
    for name in BLOCK_ORDER:
        assert name in BLOCK_TIERS, f"block {name!r} has no tier"
        assert BLOCK_TIERS[name] in {"locked", "trusted", "editable"}


def test_thoughts_toggle_tiers_present():
    """Synthetic thoughts blocks must have tiers so the dashboard can render them."""
    for name in ("thoughts_on", "thoughts_off"):
        assert name in BLOCK_TIERS


def test_locked_blocks_match_tier_set():
    assert LOCKED_BLOCKS == frozenset(
        name for name, tier in BLOCK_TIERS.items() if tier == "locked"
    )


# ---------- override semantics ------------------------------------------------


def test_editable_block_override_changes_output():
    """Overriding an editable block must change the assembled body."""
    custom = {"failure_patterns": "AVOID:\n- nothing in particular."}
    assert assemble_body(custom) != assemble_body()
    assert "AVOID:\n- nothing in particular." in assemble_body(custom)


def test_locked_block_override_is_dropped():
    """Locked-block overrides must be silently ignored to keep prompt↔parser aligned."""
    custom = {"kernel_vars": "lies about variables"}
    out = assemble_body(custom)
    assert "lies about variables" not in out
    assert out == assemble_body()


def test_unknown_block_override_is_dropped():
    """Unknown block names must be ignored, not appended to the body."""
    custom = {"some_random_key": "this should not appear in the prompt"}
    out = assemble_body(custom)
    assert "this should not appear in the prompt" not in out


def test_thoughts_override_text_used():
    """Caller may supply alternate thoughts-toggle text (dashboard preset path)."""
    out = build_system_prompt(thoughts_on=True, thoughts_on_text="\n\nCUSTOM ON.\n")
    assert out.endswith("\n\nCUSTOM ON.\n")
    assert "Thoughts mode is enabled." not in out


def test_blocks_join_with_blank_line():
    """Blocks must be separated by exactly one blank line."""
    body = assemble_body()
    # Each block boundary contributes a "\n\n" join, never "\n\n\n".
    assert "\n\n\n" not in body.rsplit("\n\n", 1)[0]


def test_json_schema_is_included():
    """ACTION_JSON_SCHEMA must be present at the tail so the agent sees the contract."""
    from gym.protocol import ACTION_JSON_SCHEMA

    body = assemble_body()
    assert ACTION_JSON_SCHEMA in body
    # Schema is last; body ends with schema + one newline.
    assert body.endswith(ACTION_JSON_SCHEMA + "\n")


# ---------- defensive: invalid input ------------------------------------------


@pytest.mark.parametrize("bad", [{}, None])
def test_empty_override_returns_default(bad):
    assert assemble_body(bad) == assemble_body()

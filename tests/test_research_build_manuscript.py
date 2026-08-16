from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.build_manuscript import ManuscriptError, build_manuscript, write_or_check


ROOT = Path(__file__).resolve().parents[1]


def test_committed_manuscript_and_manifest_are_current():
    expected_manuscript, expected_manifest = build_manuscript(ROOT)

    assert (ROOT / "paper/manuscript.md").read_bytes() == expected_manuscript
    assert (ROOT / "paper/artifact-manifest.json").read_bytes() == expected_manifest
    write_or_check(ROOT, check=True)


def test_manuscript_has_required_evidence_boundaries():
    manuscript = (ROOT / "paper/manuscript.md").read_text(encoding="utf-8")

    assert "120 terminal agent outcomes" in manuscript
    assert "did not detect a clear incremental benefit" in manuscript
    assert "makes no empirical checklist-coverage claim" in manuscript
    assert "no paid provider fallback" in manuscript.lower()
    assert "not a journal submission" in (ROOT / "paper/REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "{{GENERATED_RESULTS}}" not in manuscript


def test_manifest_binds_analysis_and_manuscript():
    manifest = json.loads((ROOT / "paper/artifact-manifest.json").read_text(encoding="utf-8"))

    assert manifest["publication_status"] == "internal_working_manuscript_not_released"
    assert manifest["analysis_result_hash"] == "22ff7ac1ffeb7ce116c384ea32ab23f82984c01aa7b16f19993b8f8f1db3c58a"
    assert "research/publication/paper_v1/primary-analysis.json" in manifest["files"]
    assert "paper/manuscript.md" in manifest["files"]


def test_check_rejects_missing_analysis_package(tmp_path):
    with pytest.raises(ManuscriptError, match="analysis file is missing"):
        write_or_check(tmp_path, check=True)

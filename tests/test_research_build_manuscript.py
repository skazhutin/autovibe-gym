from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.build_manuscript import (
    ManuscriptError,
    build_manuscript,
    canonical_text_sha256,
    validate_publication_directory,
    write_or_check,
)
from research.report_results import publication_files


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
    assert "256,000 accounted tokens in the combined sum" in manuscript
    assert "resource comparison is incomplete" in manuscript
    assert "no paid provider fallback" in manuscript.lower()
    assert "not a journal submission" in (ROOT / "paper/REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "{{GENERATED_RESULTS}}" not in manuscript


def test_manifest_binds_analysis_and_manuscript():
    manifest = json.loads((ROOT / "paper/artifact-manifest.json").read_text(encoding="utf-8"))

    assert manifest["publication_status"] == "internal_working_manuscript_not_released"
    assert manifest["file_hash_policy"] == "sha256_utf8_text_normalized_to_lf"
    assert manifest["analysis_result_hash"] == "22ff7ac1ffeb7ce116c384ea32ab23f82984c01aa7b16f19993b8f8f1db3c58a"
    assert "research/publication/paper_v1/primary-analysis.json" in manifest["files"]
    assert "research/publication/paper_v1/hidden-score-summaries.csv" in manifest["files"]
    assert "research/publication/paper_v1/SUMMARY.md" in manifest["files"]
    assert "research/report_results.py" in manifest["files"]
    assert "research/run_artifacts.py" in manifest["files"]
    assert "paper/manuscript.md" in manifest["files"]


def test_check_rejects_missing_analysis_package(tmp_path):
    with pytest.raises(ManuscriptError, match="analysis file is missing"):
        write_or_check(tmp_path, check=True)


def test_publication_validation_rejects_modified_derived_artifact(tmp_path):
    result = json.loads(
        (ROOT / "research/publication/paper_v1/primary-analysis.json").read_text(
            encoding="utf-8"
        )
    )
    for name, content in publication_files(result).items():
        (tmp_path / name).write_bytes(content)
    validate_publication_directory(result, tmp_path)

    (tmp_path / "primary-effects.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ManuscriptError, match="changed=primary-effects.csv"):
        validate_publication_directory(result, tmp_path)


def test_manifest_text_hash_is_newline_independent():
    assert canonical_text_sha256(b"first\nsecond\n") == canonical_text_sha256(
        b"first\r\nsecond\r\n"
    )

from __future__ import annotations

import json

import pytest

from research.report_results import ReportingError, publication_files, publish_results
from research.run_artifacts import canonical_hash


def _result():
    comparisons = {}
    for hypothesis, treatment, control, difference in (
        ("H1", "B", "A", -0.5),
        ("H2", "C", "B", 0.1),
    ):
        comparisons[hypothesis] = {
            "hypothesis_id": hypothesis,
            "treatment": treatment,
            "control": control,
            "pair_count": 1,
            "control_mean_fanu": 0.7,
            "treatment_mean_fanu": 0.7 + difference,
            "control_median_fanu": 0.7,
            "treatment_median_fanu": 0.7 + difference,
            "equal_stratum_mean_difference": difference,
            "median_paired_difference": difference,
            "bootstrap_ci": {"low": difference - 0.1, "high": difference + 0.1},
            "permutation": {
                "p_value": 0.05,
                "holm_adjusted_p_value": 0.1,
                "mode": "exact",
            },
            "valid_submission_mcnemar": {
                "p_value": 1.0,
                "discordant_pairs": 0,
                "control_valid_treatment_invalid": 0,
                "treatment_valid_control_invalid": 0,
            },
            "dataset_model_summaries": [
                {
                    "dataset_id": "dataset-a",
                    "model_id": "model-a",
                    "pair_count": 1,
                    "control_mean_fanu": 0.7,
                    "treatment_mean_fanu": 0.7 + difference,
                    "mean_paired_difference": difference,
                    "median_paired_difference": difference,
                }
            ],
            "pairs": [],
        }
    rows = []
    categories = {"A": "success", "B": "budget_exhausted", "C": "success"}
    for sequence, arm in enumerate(("A", "B", "C")):
        rows.append(
            {
                "sequence_index": sequence,
                "run_id": f"run-{arm}",
                "condition_id": f"condition-{arm}",
                "dataset_id": "dataset-a",
                "model_id": "model-a",
                "replicate_index": 0,
                "arm": arm,
                "failure_category": categories[arm],
                "valid_submit": arm != "B",
                "hidden_evaluations": int(arm != "B"),
                "hidden_score_observed": arm != "B",
                "analysis_score": 0.5,
                "fanu": 0.5 if arm != "B" else 0.0,
                "input_tokens": 100,
                "output_tokens": 20,
                "reasoning_tokens": 0,
                "logical_llm_calls": 2,
            }
        )
    result = {
        "schema_version": "1.0",
        "evidence_class": "confirmatory_analysis",
        "plan_id": "plan_fixture",
        "plan_hash": "1" * 64,
        "protocol_hash": "2" * 64,
        "analysis_reference_hash": "3" * 64,
        "analysis_config": {},
        "analysis_config_hash": "4" * 64,
        "reconciliation_hash": "5" * 64,
        "completeness": {
            "complete": True,
            "planned_conditions": 3,
            "terminal_agent_outcomes": 3,
        },
        "comparisons": comparisons,
        "valid_submission_rates": {
            arm: {
                "valid_submissions": int(arm != "B"),
                "total_agent_outcomes": 1,
                "rate": float(arm != "B"),
                "confidence_interval": {
                    "low": 0.0,
                    "high": 1.0,
                    "level": 0.95,
                    "method": "wilson_score",
                },
            }
            for arm in ("A", "B", "C")
        },
        "hidden_score_summaries": [
            {
                "arm": "A",
                "dataset_id": "dataset-a",
                "model_id": "model-a",
                "agent_outcomes": 1,
                "successful_outcomes": 1,
                "failure_adjusted_mean": 0.5,
                "failure_adjusted_median": 0.5,
                "successful_only_mean": 0.5,
                "successful_only_median": 0.5,
                "successful_only_is_selection_biased": True,
            }
        ],
        "rows": rows,
    }
    result["result_hash"] = canonical_hash(result)
    return result


def test_publication_files_are_deterministic_and_complete():
    first = publication_files(_result())
    second = publication_files(_result())

    assert first == second
    assert {
        "primary-analysis.json",
        "primary-effects.csv",
        "valid-submission-rates.csv",
        "failure-categories.csv",
        "resource-usage.csv",
        "stratum-effects.csv",
        "hidden-score-summaries.csv",
        "primary-effects.svg",
        "valid-submission-rates.svg",
        "failure-categories.svg",
        "resource-usage.svg",
        "SUMMARY.md",
        "provenance.json",
    } == set(first)
    assert b"B \xe2\x88\x92 A = -0.500" in first["SUMMARY.md"]
    assert b"<svg" in first["primary-effects.svg"]


def test_publication_rejects_tampered_analysis_hash():
    result = _result()
    result["rows"][0]["input_tokens"] += 1

    with pytest.raises(ReportingError, match="hash"):
        publication_files(result)


def test_publication_is_idempotent_and_refuses_overwrite(tmp_path):
    result = _result()
    created = publish_results(result, tmp_path)
    assert len(created) == 13
    assert publish_results(result, tmp_path) == []

    (tmp_path / "SUMMARY.md").write_text("changed", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_results(result, tmp_path)

    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["analysis_result_hash"] == result["result_hash"]

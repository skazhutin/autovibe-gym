import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from research.analysis import (
    AnalysisError,
    compute_fanu,
    exact_mcnemar_pvalue,
    holm_adjust,
    paired_permutation_pvalue,
    reconcile_manifest_ledgers,
    run_primary_analysis,
    valid_submission_rates,
    wilson_score_interval,
    write_analysis_result,
)
from research.planner import build_confirmatory_plan
from research.run_artifacts import canonical_hash


def _run_analysis(**kwargs):
    return run_primary_analysis(allow_synthetic_in_memory=True, **kwargs)


def _reference_payload():
    return {
        "schema_version": "1.0",
        "reference_pipeline_id": "fixture-reference-pipeline-v1",
        "datasets": [
            {
                "dataset_id": "dataset-a",
                "metric_direction": "higher",
                "dummy_score": 0.5,
                "reference_score": 0.9,
            }
        ],
    }


def _config():
    return {
        "schema_version": "1.0",
        "experiment_id": "paper-v1-analysis-fixture",
        "git_commit": "a" * 40,
        "budget_policy_hash": "b" * 64,
        "analysis_reference_hash": canonical_hash(_reference_payload()),
        "matrix": {
            "datasets": [
                {
                    "dataset_id": "dataset-a",
                    "dataset_hash": "c" * 64,
                    "split_id": "split-a",
                    "split_seed": 42,
                }
            ],
            "models": [
                {
                    "model_provider": "provider-a",
                    "model_id": "model-a",
                    "model_version": "2026-08-01",
                    "decoding_config_hash": "d" * 64,
                }
            ],
            "arms": [
                {
                    "arm": "A",
                    "product_mode": "repeated_single_shot",
                    "execution_policy_hash": "1" * 64,
                    "prompt_template_hash": "2" * 64,
                },
                {
                    "arm": "B",
                    "product_mode": "iterative_no_checklist",
                    "execution_policy_hash": "3" * 64,
                    "prompt_template_hash": "4" * 64,
                },
                {
                    "arm": "C",
                    "product_mode": "gym_with_checklist",
                    "execution_policy_hash": "5" * 64,
                    "prompt_template_hash": "6" * 64,
                },
            ],
            "replicates_per_cell": 2,
            "planned_conditions": 6,
            "randomization_seed": 20260812,
        },
    }


def _protocol():
    return {
        "status": "frozen",
        "arms": {
            "A": {"current_product_mode": "repeated_single_shot"},
            "B": {"current_product_mode": "iterative_no_checklist"},
            "C": {"current_product_mode": "gym_with_checklist"},
        },
        "matrix": {
            "status": "frozen",
            "models": [{"id": "model-a"}],
            "datasets": [{"id": "dataset-a"}],
            "confirmatory_arms": ["A", "B", "C"],
            "replicates_per_cell": 2,
            "planned_confirmatory_runs": 6,
            "block_keys": ["dataset_id", "model_id", "replicate_index"],
            "randomization_seed": 20260812,
        },
        "budget": {"status": "frozen"},
        "analysis": {
            "primary_comparison": {
                "treatment": "B",
                "control": "A",
                "outcome": "failure_adjusted_normalized_utility",
            },
            "key_secondary_comparison": {
                "treatment": "C",
                "control": "B",
                "outcome": "failure_adjusted_normalized_utility",
            },
            "paired_block_keys": ["dataset_id", "model_id", "replicate_index"],
            "aggregate_weighting": "equal_weight_per_dataset_model_stratum",
            "confidence_interval": {
                "method": "paired_stratified_percentile_bootstrap",
                "level": 0.95,
                "resamples": 200,
                "seed": 20260812,
            },
            "hypothesis_test": {
                "method": "two_sided_paired_permutation",
                "monte_carlo_draws_if_not_exact": 1000,
                "seed": 20260812,
                "familywise_correction": "holm_for_H1_and_H2_FANU_tests",
                "alpha": 0.05,
            },
            "valid_submission_test": "exact_mcnemar",
            "valid_submission_rate_interval": {
                "method": "wilson_score",
                "level": 0.95,
            },
        },
    }


def _references(plan):
    payload = _reference_payload()
    return {
        **payload,
        "reference_hash": canonical_hash(payload),
    }


def _manifest(item, run_id, *, category="success", score=0.7, rerun_of=None):
    success = category == "success"
    return {
        "schema_version": "1.0",
        "experiment_id": item["condition"]["experiment_id"],
        "condition_id": item["condition_id"],
        "run_id": run_id,
        "rerun_of": rerun_of,
        "rerun_reason": "sandbox replacement" if rerun_of else None,
        "state": "completed",
        "started_at": "2026-08-15T00:00:00Z",
        "completed_at": "2026-08-15T00:01:00Z",
        "condition": copy.deepcopy(item["condition"]),
        "provenance": {
            "git_commit": item["condition"]["git_commit"],
            "git_dirty": False,
        },
        "failure_category": category,
        "final_status": category,
        "summary": {
            "valid_submit": success,
            "hidden_evaluations": 1 if success else 0,
            **({"final_test_metric": score} if success else {}),
        },
        "ledger_totals": {
            "input_tokens": 100,
            "output_tokens": 20,
            "reasoning_tokens": 5,
            "logical_llm_calls": 2,
        },
        "artifacts": {},
    }


def _complete_fixture():
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    scores = {
        ("A", 0): 0.6,
        ("A", 1): 0.7,
        ("B", 0): 0.7,
        ("B", 1): 0.8,
        ("C", 0): 0.8,
        ("C", 1): 0.9,
    }
    manifests = []
    for index, item in enumerate(plan["conditions"]):
        condition = item["condition"]
        manifests.append(
            _manifest(
                item,
                f"run_{index:02d}",
                score=scores[(condition["arm"], condition["replicate_index"])],
            )
        )
    return plan, manifests


def test_fanu_supports_both_directions_and_does_not_clip():
    assert compute_fanu(1.1, dummy_score=0.5, reference_score=0.9, metric_direction="higher") == pytest.approx(1.5)
    assert compute_fanu(0.3, dummy_score=0.5, reference_score=0.1, metric_direction="lower") == pytest.approx(0.5)
    with pytest.raises(AnalysisError, match="denominator is zero"):
        compute_fanu(
            0.7,
            dummy_score=0.5,
            reference_score=0.5,
            metric_direction="higher",
        )


def test_complete_analysis_is_deterministic_and_paired():
    plan, manifests = _complete_fixture()
    first = _run_analysis(
        plan=plan,
        manifests=manifests,
        references=_references(plan),
        protocol=_protocol(),
    )
    second = _run_analysis(
        plan=plan,
        manifests=list(reversed(manifests)),
        references=_references(plan),
        protocol=_protocol(),
    )

    assert first == second
    assert first["completeness"]["complete"] is True
    assert first["comparisons"]["H1"]["equal_stratum_mean_difference"] == pytest.approx(0.25)
    assert first["comparisons"]["H2"]["equal_stratum_mean_difference"] == pytest.approx(0.25)
    assert first["comparisons"]["H1"]["bootstrap_ci"] == pytest.approx({"low": 0.25, "high": 0.25})
    assert first["valid_submission_rates"]["A"]["rate"] == 1.0
    assert first["valid_submission_rates"]["A"]["confidence_interval"]["method"] == "wilson_score"


def test_agent_failure_remains_at_dummy_performance():
    plan, manifests = _complete_fixture()
    target = next(item for item in plan["conditions"] if item["condition"]["arm"] == "B")
    manifests = [item for item in manifests if item["condition_id"] != target["condition_id"]]
    manifests.append(_manifest(target, "run_failed", category="invalid_submission"))

    result = _run_analysis(
        plan=plan,
        manifests=manifests,
        references=_references(plan),
        protocol=_protocol(),
    )
    row = next(item for item in result["rows"] if item["run_id"] == "run_failed")
    assert row["fanu"] == 0.0
    assert row["hidden_score_observed"] is False
    summary = next(
        item
        for item in result["hidden_score_summaries"]
        if item["arm"] == "B"
    )
    assert summary["agent_outcomes"] == 2
    assert summary["successful_outcomes"] == 1
    assert summary["successful_only_is_selection_biased"] is True


def test_dirty_or_unknown_worktree_provenance_blocks_analysis():
    plan, manifests = _complete_fixture()
    manifests[0]["provenance"]["git_dirty"] = True
    with pytest.raises(AnalysisError, match="clean-worktree"):
        _run_analysis(
            plan=plan,
            manifests=manifests,
            references=_references(plan),
            protocol=_protocol(),
        )

    manifests[0]["provenance"].pop("git_dirty")
    with pytest.raises(AnalysisError, match="clean-worktree"):
        _run_analysis(
            plan=plan,
            manifests=manifests,
            references=_references(plan),
            protocol=_protocol(),
        )


def test_disk_manifest_totals_are_recomputed_from_ledgers(tmp_path):
    usage = [
        {
            "event": "logical_llm_call",
            "input_tokens": 10,
            "output_tokens": 3,
            "reasoning_tokens": 2,
            "cached_input_tokens": 1,
        },
        {"event": "provider_request_attempt"},
    ]
    executions = [{"event": "code_execution"}]
    (tmp_path / "usage_ledger.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in usage), encoding="utf-8"
    )
    (tmp_path / "execution_ledger.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in executions), encoding="utf-8"
    )
    manifest = {
        "run_id": "run_ledger",
        "_manifest_path": str(tmp_path / "run_manifest.json"),
        "artifacts": {
            "usage_ledger": "usage_ledger.jsonl",
            "execution_ledger": "execution_ledger.jsonl",
        },
        "ledger_totals": {
            "logical_llm_calls": 1,
            "provider_request_attempts": 1,
            "input_tokens": 10,
            "output_tokens": 3,
            "reasoning_tokens": 2,
            "cached_input_tokens": 1,
            "execution_events": 1,
        },
    }

    assert reconcile_manifest_ledgers(manifest) == manifest["ledger_totals"]
    manifest["ledger_totals"]["input_tokens"] = 999
    with pytest.raises(AnalysisError, match="do not reconcile") as caught:
        reconcile_manifest_ledgers(manifest)
    assert caught.value.details["code"] == "ledger_totals_mismatch"


def test_incomplete_run_set_blocks_effects():
    plan, manifests = _complete_fixture()
    with pytest.raises(AnalysisError, match="incomplete") as caught:
        _run_analysis(
            plan=plan,
            manifests=manifests[:-1],
            references=_references(plan),
            protocol=_protocol(),
        )
    assert caught.value.details["code"] == "incomplete_run_set"
    assert caught.value.details["counts"]["pending"] == 1


@pytest.mark.parametrize(
    ("summary_patch", "message"),
    [
        ({"valid_submit": False}, "valid_submit=true"),
        ({"hidden_evaluations": 0}, "exactly one hidden"),
    ],
)
def test_success_requires_valid_submit_and_exactly_one_hidden_evaluation(summary_patch, message):
    plan, manifests = _complete_fixture()
    manifests[0]["summary"].update(summary_patch)
    with pytest.raises(AnalysisError, match=message):
        _run_analysis(
            plan=plan,
            manifests=manifests,
            references=_references(plan),
            protocol=_protocol(),
        )


def test_reference_payload_and_denominator_are_gates():
    plan, manifests = _complete_fixture()
    references = _references(plan)
    references["reference_pipeline_id"] = "changed-after-freeze"
    with pytest.raises(AnalysisError, match="reference_hash"):
        _run_analysis(plan=plan, manifests=manifests, references=references, protocol=_protocol())

    references = _references(plan)
    references["datasets"][0]["reference_score"] = 0.5
    with pytest.raises(AnalysisError, match="equals dummy"):
        _run_analysis(plan=plan, manifests=manifests, references=references, protocol=_protocol())


def test_reference_values_and_protocol_are_bound_to_the_plan():
    plan, manifests = _complete_fixture()
    references = _references(plan)
    references["datasets"][0]["reference_score"] = 0.85
    with pytest.raises(AnalysisError, match="reference_hash"):
        _run_analysis(
            plan=plan,
            manifests=manifests,
            references=references,
            protocol=_protocol(),
        )

    references["reference_hash"] = canonical_hash(
        {
            "schema_version": "1.0",
            "reference_pipeline_id": references["reference_pipeline_id"],
            "datasets": references["datasets"],
        }
    )
    with pytest.raises(AnalysisError, match="immutable plan"):
        _run_analysis(
            plan=plan,
            manifests=manifests,
            references=references,
            protocol=_protocol(),
        )

    changed_protocol = _protocol()
    changed_protocol["analysis"]["confidence_interval"]["seed"] = 99
    with pytest.raises(AnalysisError, match="protocol hash"):
        _run_analysis(
            plan=plan,
            manifests=manifests,
            references=_references(plan),
            protocol=changed_protocol,
        )


def test_exact_permutation_mcnemar_and_holm_helpers():
    pairs = [
        {"dataset_id": "d", "model_id": "m", "difference": 0.25},
        {"dataset_id": "d", "model_id": "m", "difference": 0.25},
    ]
    pvalue, mode = paired_permutation_pvalue(
        pairs,
        exact_max_nonzero_pairs=20,
        monte_carlo_draws=100,
        seed=1,
    )
    assert (pvalue, mode) == (0.5, "exact")

    paired_validity = [
        {"treatment_valid": True, "control_valid": False},
        {"treatment_valid": True, "control_valid": False},
        {"treatment_valid": False, "control_valid": True},
    ]
    assert exact_mcnemar_pvalue(paired_validity)["p_value"] == 1.0
    assert holm_adjust({"H1": 0.01, "H2": 0.04}) == {"H1": 0.02, "H2": 0.04}


def test_wilson_rate_intervals_are_preregistered_per_arm():
    low, high = wilson_score_interval(1, 2, level=0.95)
    assert 0.0 < low < 0.5 < high < 1.0
    rows = [
        {"arm": arm, "valid_submit": replicate == 0}
        for arm in ("A", "B", "C")
        for replicate in range(2)
    ]
    rates = valid_submission_rates(rows, level=0.95)
    assert set(rates) == {"A", "B", "C"}
    assert all(item["rate"] == 0.5 for item in rates.values())


def test_result_publish_is_idempotent_and_concurrent_safe(tmp_path):
    result = {"schema_version": "1.0", "result_hash": "abc"}
    path = tmp_path / "analysis.json"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: write_analysis_result(result, path), range(2)))

    assert sorted(outcomes) == [False, True]
    assert write_analysis_result(result, path) is False
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_analysis_result({**result, "result_hash": "different"}, path)
    assert not list(tmp_path.glob(".*.tmp"))

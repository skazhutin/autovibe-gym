"""Preregistered Paper V1 primary analysis over immutable run manifests."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from research.planner import (
    ARM_IDS,
    BLOCK_KEYS,
    INFRASTRUCTURE_FAILURES,
    PlanError,
    discover_manifests,
    load_plan,
    reconcile_plan,
)
from research.run_artifacts import FailureCategory, canonical_hash, validate_manifest


ANALYSIS_SCHEMA_VERSION = "1.0"
AGENT_OUTCOME_FAILURES = {
    FailureCategory.INVALID_SUBMISSION.value,
    FailureCategory.AGENT_CODE_FAILURE.value,
    FailureCategory.BUDGET_EXHAUSTED.value,
    FailureCategory.EXECUTION_TIMEOUT.value,
}
PRIMARY_COMPARISONS = (
    ("H1", "B", "A"),
    ("H2", "C", "B"),
)


class AnalysisError(ValueError):
    """Raised when immutable inputs do not satisfy the preregistered analysis."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.details = dict(details or {})


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{label} must be a mapping")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise AnalysisError(f"{label} must be finite")
    return number


def analysis_config_from_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    analysis = _mapping(protocol.get("analysis"), "protocol.analysis")
    confidence = _mapping(analysis.get("confidence_interval"), "confidence_interval")
    hypothesis = _mapping(analysis.get("hypothesis_test"), "hypothesis_test")
    expected = {
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
        "paired_block_keys": list(BLOCK_KEYS),
        "aggregate_weighting": "equal_weight_per_dataset_model_stratum",
        "confidence_method": "paired_stratified_percentile_bootstrap",
        "hypothesis_method": "two_sided_paired_permutation",
        "familywise_correction": "holm_for_H1_and_H2_FANU_tests",
        "valid_submission_test": "exact_mcnemar",
    }
    actual = {
        "primary_comparison": analysis.get("primary_comparison"),
        "key_secondary_comparison": analysis.get("key_secondary_comparison"),
        "paired_block_keys": analysis.get("paired_block_keys"),
        "aggregate_weighting": analysis.get("aggregate_weighting"),
        "confidence_method": confidence.get("method"),
        "hypothesis_method": hypothesis.get("method"),
        "familywise_correction": hypothesis.get("familywise_correction"),
        "valid_submission_test": analysis.get("valid_submission_test"),
    }
    if actual != expected:
        raise AnalysisError("protocol.analysis does not match the preregistered pipeline contract")

    level = _number(confidence.get("level"), "confidence_interval.level")
    resamples = int(confidence.get("resamples") or 0)
    bootstrap_seed = int(confidence.get("seed") or 0)
    permutation_draws = int(hypothesis.get("monte_carlo_draws_if_not_exact") or 0)
    permutation_seed = int(hypothesis.get("seed") or 0)
    alpha = _number(hypothesis.get("alpha"), "hypothesis_test.alpha")
    if not 0 < level < 1 or resamples <= 0 or permutation_draws <= 0 or not 0 < alpha < 1:
        raise AnalysisError("protocol analysis resampling, confidence, and alpha values are invalid")
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "paired_block_keys": list(BLOCK_KEYS),
        "aggregate_weighting": expected["aggregate_weighting"],
        "confidence_interval": {
            "method": expected["confidence_method"],
            "level": level,
            "resamples": resamples,
            "seed": bootstrap_seed,
        },
        "hypothesis_test": {
            "method": expected["hypothesis_method"],
            "exact_max_nonzero_pairs": 20,
            "monte_carlo_draws": permutation_draws,
            "seed": permutation_seed,
            "monte_carlo_pvalue": "add_one_correction",
            "familywise_correction": expected["familywise_correction"],
            "alpha": alpha,
        },
        "valid_submission_test": expected["valid_submission_test"],
    }


def validate_references(
    references: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    references = dict(_mapping(references, "references"))
    if references.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise AnalysisError("references.schema_version must be '1.0'")
    if references.get("plan_id") != plan.get("plan_id"):
        raise AnalysisError("references.plan_id does not match the immutable plan")
    if references.get("plan_hash") != plan.get("plan_hash"):
        raise AnalysisError("references.plan_hash does not match the immutable plan")
    items = references.get("datasets")
    if not isinstance(items, list) or not items:
        raise AnalysisError("references.datasets must be a non-empty list")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in items:
        item = dict(_mapping(raw, "dataset reference"))
        expected_keys = {
            "dataset_id",
            "metric_direction",
            "dummy_score",
            "reference_score",
        }
        if set(item) != expected_keys:
            raise AnalysisError("dataset reference fields do not match the frozen schema")
        dataset_id = str(item.get("dataset_id") or "")
        if not dataset_id or dataset_id in by_id:
            raise AnalysisError("dataset references must have unique non-empty dataset_id values")
        direction = str(item.get("metric_direction") or "")
        if direction not in {"higher", "lower"}:
            raise AnalysisError(f"dataset {dataset_id} metric_direction must be higher or lower")
        dummy = _number(item.get("dummy_score"), f"dataset {dataset_id} dummy_score")
        reference = _number(item.get("reference_score"), f"dataset {dataset_id} reference_score")
        if reference == dummy:
            raise AnalysisError(f"dataset {dataset_id} reference_score equals dummy_score")
        if direction == "higher" and reference < dummy:
            raise AnalysisError(f"dataset {dataset_id} higher-is-better reference is worse than dummy")
        if direction == "lower" and reference > dummy:
            raise AnalysisError(f"dataset {dataset_id} lower-is-better reference is worse than dummy")
        item["dummy_score"] = dummy
        item["reference_score"] = reference
        by_id[dataset_id] = item
    planned_ids = set(plan["matrix"]["dataset_ids"])
    if set(by_id) != planned_ids:
        raise AnalysisError("dataset references do not match the immutable plan dataset IDs")
    reference_payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "datasets": [by_id[dataset_id] for dataset_id in sorted(by_id)],
    }
    reference_hash = canonical_hash(reference_payload)
    if references.get("reference_hash") != reference_hash:
        raise AnalysisError("references.reference_hash does not match the canonical payload")
    if plan.get("analysis_reference_hash") != reference_hash:
        raise AnalysisError("frozen reference hash does not match the immutable plan")
    return by_id


def compute_fanu(
    score: float,
    *,
    dummy_score: float,
    reference_score: float,
    metric_direction: str,
) -> float:
    score = _number(score, "score")
    dummy_score = _number(dummy_score, "dummy_score")
    reference_score = _number(reference_score, "reference_score")
    if reference_score == dummy_score:
        raise AnalysisError("FANU denominator is zero")
    if metric_direction == "higher":
        return (score - dummy_score) / (reference_score - dummy_score)
    if metric_direction == "lower":
        return (dummy_score - score) / (dummy_score - reference_score)
    raise AnalysisError("metric_direction must be higher or lower")


def _hidden_score(manifest: Mapping[str, Any]) -> float:
    summary = _mapping(manifest.get("summary"), "manifest.summary")
    for key in ("final_test_metric", "test_metric"):
        if summary.get(key) is not None:
            return _number(summary[key], f"manifest.summary.{key}")
    raise AnalysisError(f"successful run {manifest.get('run_id')} is missing its hidden score")


def build_analysis_rows(
    plan: Mapping[str, Any],
    manifests: Iterable[Mapping[str, Any]],
    references: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifests = [dict(item) for item in manifests]
    for manifest in manifests:
        validate_manifest(manifest)
    try:
        reconciliation = reconcile_plan(plan, manifests)
    except PlanError as exc:
        raise AnalysisError(str(exc)) from exc
    if not reconciliation["complete"]:
        raise AnalysisError(
            "confirmatory run set is incomplete; arm effects are blocked",
            details={
                "code": "incomplete_run_set",
                "plan_id": plan.get("plan_id"),
                "plan_hash": plan.get("plan_hash"),
                "counts": reconciliation["counts"],
                "conditions": reconciliation["conditions"],
                "reconciliation_hash": reconciliation["reconciliation_hash"],
            },
        )
    reference_by_dataset = validate_references(references, plan=plan)
    by_run_id = {str(item["run_id"]): item for item in manifests}
    status_by_condition = {
        item["condition_id"]: item for item in reconciliation["conditions"]
    }
    rows: list[dict[str, Any]] = []
    for planned in plan["conditions"]:
        status = status_by_condition[planned["condition_id"]]
        manifest = by_run_id[str(status["latest_run_id"])]
        category = str(manifest["failure_category"])
        dataset_id = str(planned["condition"]["dataset_id"])
        reference = reference_by_dataset[dataset_id]
        summary = _mapping(manifest.get("summary"), "manifest.summary")
        hidden_evaluations = int(summary.get("hidden_evaluations") or 0)
        if category == FailureCategory.SUCCESS.value:
            if summary.get("valid_submit") is not True:
                raise AnalysisError(
                    f"successful run {manifest.get('run_id')} must declare valid_submit=true"
                )
            if hidden_evaluations != 1:
                raise AnalysisError(
                    f"successful run {manifest.get('run_id')} must have exactly one hidden evaluation"
                )
            score = _hidden_score(manifest)
            valid_submit = True
        elif category in AGENT_OUTCOME_FAILURES:
            if summary.get("valid_submit") is True:
                raise AnalysisError(
                    f"failed run {manifest.get('run_id')} cannot declare valid_submit=true"
                )
            if hidden_evaluations > 1:
                raise AnalysisError(
                    f"failed run {manifest.get('run_id')} exceeds the one hidden evaluation gate"
                )
            score = float(reference["dummy_score"])
            valid_submit = False
        elif category in INFRASTRUCTURE_FAILURES:
            raise AnalysisError("infrastructure attempt reached terminal analysis population")
        else:
            raise AnalysisError(f"unsupported terminal failure category: {category}")
        condition = planned["condition"]
        ledger = _mapping(manifest.get("ledger_totals"), "manifest.ledger_totals")
        rows.append(
            {
                "sequence_index": planned["sequence_index"],
                "condition_id": planned["condition_id"],
                "run_id": manifest["run_id"],
                "dataset_id": dataset_id,
                "model_id": condition["model_id"],
                "replicate_index": int(condition["replicate_index"]),
                "arm": condition["arm"],
                "failure_category": category,
                "valid_submit": valid_submit,
                "hidden_score_observed": category == FailureCategory.SUCCESS.value,
                "analysis_score": score,
                "fanu": compute_fanu(
                    score,
                    dummy_score=reference["dummy_score"],
                    reference_score=reference["reference_score"],
                    metric_direction=reference["metric_direction"],
                ),
                "input_tokens": int(ledger.get("input_tokens") or 0),
                "output_tokens": int(ledger.get("output_tokens") or 0),
                "reasoning_tokens": int(ledger.get("reasoning_tokens") or 0),
                "logical_llm_calls": int(ledger.get("logical_llm_calls") or 0),
                "hidden_evaluations": hidden_evaluations,
            }
        )
    return rows, reconciliation


def completeness_table(
    plan: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    planned_by_arm = Counter(item["condition"]["arm"] for item in plan["conditions"])
    terminal_by_arm = Counter(item["arm"] for item in rows)
    failures = Counter((item["arm"], item["failure_category"]) for item in rows)
    infrastructure_attempts = Counter(
        str(item.get("failure_category"))
        for item in manifests
        if item.get("state") == "completed"
        and item.get("failure_category") in INFRASTRUCTURE_FAILURES
    )
    return {
        "planned_conditions": len(plan["conditions"]),
        "terminal_agent_outcomes": len(rows),
        "total_attempts": len(manifests),
        "planned_by_arm": {arm: planned_by_arm[arm] for arm in ARM_IDS},
        "terminal_by_arm": {arm: terminal_by_arm[arm] for arm in ARM_IDS},
        "agent_outcomes_by_arm_and_category": [
            {"arm": arm, "failure_category": category, "count": count}
            for (arm, category), count in sorted(failures.items())
        ],
        "infrastructure_attempts_by_category": dict(sorted(infrastructure_attempts.items())),
        "complete": len(rows) == len(plan["conditions"]),
    }


def paired_rows(
    rows: Sequence[Mapping[str, Any]], *, treatment: str, control: str
) -> list[dict[str, Any]]:
    by_block: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        block = tuple(row[key] for key in BLOCK_KEYS)
        arm = str(row["arm"])
        if arm in by_block[block]:
            raise AnalysisError(f"duplicate arm {arm} in paired block {block}")
        by_block[block][arm] = row
    pairs = []
    for block in sorted(by_block):
        arm_rows = by_block[block]
        if treatment not in arm_rows or control not in arm_rows:
            raise AnalysisError(f"paired block {block} is missing {treatment} or {control}")
        treatment_row = arm_rows[treatment]
        control_row = arm_rows[control]
        pairs.append(
            {
                "dataset_id": block[0],
                "model_id": block[1],
                "replicate_index": block[2],
                "treatment_arm": treatment,
                "control_arm": control,
                "treatment_fanu": float(treatment_row["fanu"]),
                "control_fanu": float(control_row["fanu"]),
                "difference": float(treatment_row["fanu"]) - float(control_row["fanu"]),
                "treatment_valid": bool(treatment_row["valid_submit"]),
                "control_valid": bool(control_row["valid_submit"]),
            }
        )
    return pairs


def equal_stratum_mean(pairs: Sequence[Mapping[str, Any]], key: str = "difference") -> float:
    strata: dict[tuple[str, str], list[float]] = defaultdict(list)
    for pair in pairs:
        strata[(str(pair["dataset_id"]), str(pair["model_id"]))].append(float(pair[key]))
    if not strata:
        raise AnalysisError("paired analysis has no strata")
    return mean(mean(values) for values in strata.values())


def paired_stratified_bootstrap_ci(
    pairs: Sequence[Mapping[str, Any]],
    *,
    level: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    strata: dict[tuple[str, str], np.ndarray] = defaultdict(list)
    for pair in pairs:
        strata[(str(pair["dataset_id"]), str(pair["model_id"]))].append(
            float(pair["difference"])
        )
    arrays = [np.asarray(strata[key], dtype=float) for key in sorted(strata)]
    if not arrays or resamples <= 0:
        raise AnalysisError("bootstrap requires non-empty pairs and positive resamples")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        stratum_means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for values in arrays]
        estimates[index] = float(np.mean(stratum_means))
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return float(low), float(high)


def paired_permutation_pvalue(
    pairs: Sequence[Mapping[str, Any]],
    *,
    exact_max_nonzero_pairs: int,
    monte_carlo_draws: int,
    seed: int,
) -> tuple[float, str]:
    values = np.asarray([float(pair["difference"]) for pair in pairs], dtype=float)
    nonzero_indices = np.flatnonzero(values != 0.0)
    if len(nonzero_indices) == 0:
        return 1.0, "exact"
    strata = [(str(pair["dataset_id"]), str(pair["model_id"])) for pair in pairs]

    def statistic(signed_values: np.ndarray) -> float:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for stratum, value in zip(strata, signed_values):
            grouped[stratum].append(float(value))
        return abs(mean(mean(items) for items in grouped.values()))

    observed = statistic(values)
    tolerance = 1e-15
    if len(nonzero_indices) <= exact_max_nonzero_pairs:
        total = 2 ** len(nonzero_indices)
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero_indices)):
            signed = values.copy()
            signed[nonzero_indices] *= np.asarray(signs)
            permuted = statistic(signed)
            extreme += permuted + tolerance >= observed
        return extreme / total, "exact"
    if monte_carlo_draws <= 0:
        raise AnalysisError("Monte Carlo permutation requires positive draws")
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = monte_carlo_draws
    while remaining:
        batch = min(10_000, remaining)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]), size=(batch, len(nonzero_indices))
        )
        for row in signs:
            signed = values.copy()
            signed[nonzero_indices] *= row
            extreme += statistic(signed) + tolerance >= observed
        remaining -= batch
    return (extreme + 1) / (monte_carlo_draws + 1), "monte_carlo_add_one"


def exact_mcnemar_pvalue(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    treatment_only = sum(
        bool(pair["treatment_valid"]) and not bool(pair["control_valid"]) for pair in pairs
    )
    control_only = sum(
        bool(pair["control_valid"]) and not bool(pair["treatment_valid"]) for pair in pairs
    )
    discordant = treatment_only + control_only
    if discordant == 0:
        pvalue = 1.0
    else:
        lower = min(treatment_only, control_only)
        cumulative = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2**discordant)
        pvalue = min(1.0, 2.0 * cumulative)
    return {
        "treatment_valid_control_invalid": treatment_only,
        "control_valid_treatment_invalid": control_only,
        "discordant_pairs": discordant,
        "p_value": pvalue,
    }


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted((float(value), key) for key, value in pvalues.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (value, key) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[key] = running
    return adjusted


def analyze_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    hypothesis_id: str,
    treatment: str,
    control: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    pairs = paired_rows(rows, treatment=treatment, control=control)
    confidence = config["confidence_interval"]
    hypothesis = config["hypothesis_test"]
    ci_low, ci_high = paired_stratified_bootstrap_ci(
        pairs,
        level=float(confidence["level"]),
        resamples=int(confidence["resamples"]),
        seed=int(confidence["seed"]),
    )
    pvalue, permutation_mode = paired_permutation_pvalue(
        pairs,
        exact_max_nonzero_pairs=int(hypothesis["exact_max_nonzero_pairs"]),
        monte_carlo_draws=int(hypothesis["monte_carlo_draws"]),
        seed=int(hypothesis["seed"]),
    )
    treatment_values = [float(pair["treatment_fanu"]) for pair in pairs]
    control_values = [float(pair["control_fanu"]) for pair in pairs]
    differences = [float(pair["difference"]) for pair in pairs]
    stratum_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        stratum_groups[(str(pair["dataset_id"]), str(pair["model_id"]))].append(pair)
    stratum_summaries = [
        {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "pair_count": len(items),
            "treatment_mean_fanu": mean(float(item["treatment_fanu"]) for item in items),
            "control_mean_fanu": mean(float(item["control_fanu"]) for item in items),
            "mean_paired_difference": mean(float(item["difference"]) for item in items),
            "median_paired_difference": median(float(item["difference"]) for item in items),
        }
        for (dataset_id, model_id), items in sorted(stratum_groups.items())
    ]
    return {
        "hypothesis_id": hypothesis_id,
        "treatment": treatment,
        "control": control,
        "pair_count": len(pairs),
        "treatment_mean_fanu": equal_stratum_mean(pairs, key="treatment_fanu"),
        "control_mean_fanu": equal_stratum_mean(pairs, key="control_fanu"),
        "treatment_median_fanu": median(treatment_values),
        "control_median_fanu": median(control_values),
        "equal_stratum_mean_difference": equal_stratum_mean(pairs),
        "median_paired_difference": median(differences),
        "bootstrap_ci": {"low": ci_low, "high": ci_high},
        "permutation": {"p_value": pvalue, "mode": permutation_mode},
        "valid_submission_mcnemar": exact_mcnemar_pvalue(pairs),
        "dataset_model_summaries": stratum_summaries,
        "pairs": pairs,
    }


def run_primary_analysis(
    *,
    plan: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
    references: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    protocol_hash = canonical_hash(protocol)
    if plan.get("protocol_hash") != protocol_hash:
        raise AnalysisError("protocol hash does not match the immutable plan")
    config = analysis_config_from_protocol(protocol)
    rows, reconciliation = build_analysis_rows(plan, manifests, references)
    completeness = completeness_table(plan, manifests, rows)
    comparisons = {
        hypothesis_id: analyze_comparison(
            rows,
            hypothesis_id=hypothesis_id,
            treatment=treatment,
            control=control,
            config=config,
        )
        for hypothesis_id, treatment, control in PRIMARY_COMPARISONS
    }
    raw_pvalues = {
        hypothesis_id: result["permutation"]["p_value"]
        for hypothesis_id, result in comparisons.items()
    }
    adjusted = holm_adjust(raw_pvalues)
    for hypothesis_id, value in adjusted.items():
        comparisons[hypothesis_id]["permutation"]["holm_adjusted_p_value"] = value
    result: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "evidence_class": "confirmatory_analysis",
        "plan_id": plan["plan_id"],
        "plan_hash": plan["plan_hash"],
        "protocol_hash": protocol_hash,
        "analysis_reference_hash": plan["analysis_reference_hash"],
        "analysis_config": config,
        "analysis_config_hash": canonical_hash(config),
        "completeness": completeness,
        "reconciliation_hash": reconciliation["reconciliation_hash"],
        "rows": rows,
        "comparisons": comparisons,
    }
    result["result_hash"] = canonical_hash(result)
    return result


def write_analysis_result(result: Mapping[str, Any], path: str | Path) -> bool:
    """Publish a complete result once; identical reruns are idempotent."""
    output = Path(path)
    encoded = (
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
            return True
        except FileExistsError:
            if output.read_bytes() == encoded:
                return False
            raise FileExistsError(f"Refusing to overwrite analysis output: {output}")
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if source.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    return dict(_mapping(value, str(source)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper V1 preregistered primary analysis")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument(
        "--protocol", default=str(Path(__file__).parent / "protocols" / "protocol_v1.yaml")
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run_primary_analysis(
            plan=load_plan(args.plan),
            manifests=discover_manifests(args.runs),
            references=_load_mapping(args.references),
            protocol=_load_mapping(args.protocol),
        )
    except (AnalysisError, PlanError, ValueError) as exc:
        error = {
            "error": "analysis_validation_failed",
            "message": str(exc),
            "details": getattr(exc, "details", {}),
        }
        print(json.dumps(error, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from exc
    output = Path(args.output)
    created = write_analysis_result(result, output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "result_hash": result["result_hash"],
                "created": created,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

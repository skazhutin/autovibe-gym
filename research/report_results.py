"""Deterministically render Paper V1 publication artifacts from primary analysis."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from research.run_artifacts import canonical_hash


ARM_LABELS = {
    "A": "A: budget-matched best-of-N",
    "B": "B: iterative feedback",
    "C": "C: feedback + checklist",
}
HIDDEN_SUMMARY_FIELDS = (
    "arm",
    "dataset_id",
    "model_id",
    "agent_outcomes",
    "successful_outcomes",
    "failure_adjusted_mean",
    "failure_adjusted_median",
    "successful_only_mean",
    "successful_only_median",
    "successful_only_is_selection_biased",
)


class ReportingError(ValueError):
    """Raised when analysis input or publication output is not reproducible."""


def _fmt(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _csv(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def _validate_result(result: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(result)
    required = {
        "schema_version",
        "evidence_class",
        "plan_id",
        "plan_hash",
        "protocol_hash",
        "analysis_reference_hash",
        "analysis_config_hash",
        "reconciliation_hash",
        "result_hash",
        "completeness",
        "comparisons",
        "valid_submission_rates",
        "hidden_score_summaries",
        "rows",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ReportingError(f"analysis result is missing fields: {', '.join(missing)}")
    if result["evidence_class"] != "confirmatory_analysis":
        raise ReportingError("analysis result is not confirmatory evidence")
    if not result["completeness"].get("complete"):
        raise ReportingError("analysis result did not pass the completeness gate")
    claimed_hash = result["result_hash"]
    payload = dict(result)
    payload.pop("result_hash")
    if canonical_hash(payload) != claimed_hash:
        raise ReportingError("analysis result hash does not match its canonical payload")
    if set(result["comparisons"]) != {"H1", "H2"}:
        raise ReportingError("analysis result must contain exactly H1 and H2")
    if set(result["valid_submission_rates"]) != {"A", "B", "C"}:
        raise ReportingError("analysis result must contain A/B/C valid-submission rates")
    return result


def _primary_effect_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hypothesis_id in ("H1", "H2"):
        comparison = result["comparisons"][hypothesis_id]
        rows.append(
            {
                "hypothesis": hypothesis_id,
                "contrast": f"{comparison['treatment']} - {comparison['control']}",
                "pairs": comparison["pair_count"],
                "control_mean_fanu": _fmt(comparison["control_mean_fanu"]),
                "treatment_mean_fanu": _fmt(comparison["treatment_mean_fanu"]),
                "mean_difference": _fmt(comparison["equal_stratum_mean_difference"]),
                "ci95_low": _fmt(comparison["bootstrap_ci"]["low"]),
                "ci95_high": _fmt(comparison["bootstrap_ci"]["high"]),
                "median_difference": _fmt(comparison["median_paired_difference"]),
                "permutation_p": _fmt(comparison["permutation"]["p_value"]),
                "holm_p": _fmt(comparison["permutation"]["holm_adjusted_p_value"]),
                "permutation_mode": comparison["permutation"]["mode"],
                "mcnemar_p": _fmt(comparison["valid_submission_mcnemar"]["p_value"]),
            }
        )
    return rows


def _valid_rate_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for arm in ("A", "B", "C"):
        rate = result["valid_submission_rates"][arm]
        rows.append(
            {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "valid_submissions": rate["valid_submissions"],
                "total_outcomes": rate["total_agent_outcomes"],
                "rate": _fmt(rate["rate"]),
                "ci95_low": _fmt(rate["confidence_interval"]["low"]),
                "ci95_high": _fmt(rate["confidence_interval"]["high"]),
                "interval_method": rate["confidence_interval"]["method"],
            }
        )
    return rows


def _failure_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    counts = Counter((row["arm"], row["failure_category"]) for row in result["rows"])
    categories = sorted({category for _, category in counts})
    return [
        {"arm": arm, "failure_category": category, "count": counts[(arm, category)]}
        for arm in ("A", "B", "C")
        for category in categories
        if counts[(arm, category)]
    ]


def _resource_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for arm in ("A", "B", "C"):
        arm_rows = [row for row in result["rows"] if row["arm"] == arm]
        rows.append(
            {
                "arm": arm,
                "outcomes": len(arm_rows),
                "input_tokens": sum(row["input_tokens"] for row in arm_rows),
                "output_tokens": sum(row["output_tokens"] for row in arm_rows),
                "reasoning_tokens": sum(row["reasoning_tokens"] for row in arm_rows),
                "logical_llm_calls": sum(row["logical_llm_calls"] for row in arm_rows),
                "mean_protocol_tokens_per_outcome": _fmt(
                    sum(
                        row["input_tokens"]
                        + row["output_tokens"]
                        + row["reasoning_tokens"]
                        for row in arm_rows
                    )
                    / len(arm_rows),
                    2,
                ),
            }
        )
    return rows


def _stratum_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for hypothesis_id in ("H1", "H2"):
        for item in result["comparisons"][hypothesis_id]["dataset_model_summaries"]:
            rows.append(
                {
                    "hypothesis": hypothesis_id,
                    "dataset_id": item["dataset_id"],
                    "model_id": item["model_id"],
                    "pairs": item["pair_count"],
                    "control_mean_fanu": _fmt(item["control_mean_fanu"]),
                    "treatment_mean_fanu": _fmt(item["treatment_mean_fanu"]),
                    "mean_difference": _fmt(item["mean_paired_difference"]),
                    "median_difference": _fmt(item["median_paired_difference"]),
                }
            )
    return rows


def _svg_shell(title: str, body: str, *, height: int = 420) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" '
        f'viewBox="0 0 900 {height}" role="img" aria-label="{escape(title)}">\n'
        '<rect width="900" height="100%" fill="#ffffff"/>\n'
        f'<text x="60" y="42" font-family="Arial, sans-serif" font-size="22" '
        f'font-weight="700" fill="#172033">{escape(title)}</text>\n'
        f'{body}</svg>\n'
    ).encode("utf-8")


def _effect_svg(result: Mapping[str, Any]) -> bytes:
    items = [result["comparisons"][key] for key in ("H1", "H2")]
    low = min(0.0, *(item["bootstrap_ci"]["low"] for item in items))
    high = max(0.0, *(item["bootstrap_ci"]["high"] for item in items))
    padding = max((high - low) * 0.08, 0.05)
    low -= padding
    high += padding
    x0, x1 = 260.0, 840.0
    scale = lambda value: x0 + (float(value) - low) / (high - low) * (x1 - x0)
    parts = [
        f'<line x1="{scale(0):.2f}" y1="75" x2="{scale(0):.2f}" y2="300" stroke="#6b7280" stroke-dasharray="5 5"/>',
    ]
    for index, (hypothesis, item) in enumerate(zip(("H1", "H2"), items)):
        y = 125 + index * 105
        estimate = item["equal_stratum_mean_difference"]
        parts.extend(
            [
                f'<text x="60" y="{y + 6}" font-family="Arial, sans-serif" font-size="17" fill="#172033">{hypothesis}: {item["treatment"]} − {item["control"]}</text>',
                f'<line x1="{scale(item["bootstrap_ci"]["low"]):.2f}" y1="{y}" x2="{scale(item["bootstrap_ci"]["high"]):.2f}" y2="{y}" stroke="#2563eb" stroke-width="5"/>',
                f'<circle cx="{scale(estimate):.2f}" cy="{y}" r="9" fill="#0f3d91"/>',
                f'<text x="260" y="{y + 34}" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">Δ={estimate:.3f}; 95% CI [{item["bootstrap_ci"]["low"]:.3f}, {item["bootstrap_ci"]["high"]:.3f}]</text>',
            ]
        )
    parts.append(
        f'<text x="{scale(0):.2f}" y="340" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">no difference</text>'
    )
    parts.append('<text x="550" y="385" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#172033">Equal-stratum mean FANU difference</text>')
    return _svg_shell("Primary and key-secondary effects", "\n".join(parts))


def _rate_svg(result: Mapping[str, Any]) -> bytes:
    parts = []
    for index, arm in enumerate(("A", "B", "C")):
        item = result["valid_submission_rates"][arm]
        x = 185 + index * 255
        base, top = 335.0, 75.0
        height = item["rate"] * (base - top)
        ci_low = base - item["confidence_interval"]["low"] * (base - top)
        ci_high = base - item["confidence_interval"]["high"] * (base - top)
        parts.extend(
            [
                f'<rect x="{x}" y="{base-height:.2f}" width="110" height="{height:.2f}" fill="#2563eb"/>',
                f'<line x1="{x+55}" y1="{ci_high:.2f}" x2="{x+55}" y2="{ci_low:.2f}" stroke="#111827" stroke-width="3"/>',
                f'<line x1="{x+40}" y1="{ci_high:.2f}" x2="{x+70}" y2="{ci_high:.2f}" stroke="#111827" stroke-width="3"/>',
                f'<line x1="{x+40}" y1="{ci_low:.2f}" x2="{x+70}" y2="{ci_low:.2f}" stroke="#111827" stroke-width="3"/>',
                f'<text x="{x+55}" y="{base-height-12:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#172033">{item["rate"]:.1%}</text>',
                f'<text x="{x+55}" y="365" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#172033">Arm {arm}</text>',
            ]
        )
    parts.extend(
        [
            '<line x1="115" y1="335" x2="825" y2="335" stroke="#111827"/>',
            '<line x1="115" y1="75" x2="115" y2="335" stroke="#111827"/>',
            '<text x="65" y="210" transform="rotate(-90 65 210)" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#172033">Valid submission rate</text>',
        ]
    )
    return _svg_shell("Valid submissions with 95% Wilson intervals", "\n".join(parts))


def _failure_svg(result: Mapping[str, Any]) -> bytes:
    counts = Counter((row["arm"], row["failure_category"]) for row in result["rows"])
    categories = sorted({category for _, category in counts})
    semantic_colors = {
        "success": "#16a34a",
        "invalid_submission": "#f59e0b",
        "budget_exhausted": "#dc2626",
    }
    colors = {
        category: semantic_colors.get(category, "#7c3aed") for category in categories
    }
    parts = []
    for index, arm in enumerate(("A", "B", "C")):
        x, y = 175 + index * 240, 335.0
        for category in categories:
            count = counts[(arm, category)]
            height = count / 40 * 250
            y -= height
            parts.append(f'<rect x="{x}" y="{y:.2f}" width="120" height="{height:.2f}" fill="{colors[category]}"/>')
        parts.append(f'<text x="{x+60}" y="365" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#172033">Arm {arm}</text>')
    for index, category in enumerate(categories):
        x = 100 + index * 205
        parts.extend(
            [
                f'<rect x="{x}" y="390" width="16" height="16" fill="{colors[category]}"/>',
                f'<text x="{x+23}" y="404" font-family="Arial, sans-serif" font-size="13" fill="#172033">{escape(category)}</text>',
            ]
        )
    return _svg_shell("Terminal outcomes by arm", "\n".join(parts), height=440)


def _resource_svg(result: Mapping[str, Any]) -> bytes:
    resources = _resource_rows(result)
    maximum = max(
        row["input_tokens"] + row["output_tokens"] + row["reasoning_tokens"]
        for row in resources
    )
    parts = []
    for index, row in enumerate(resources):
        y = 105 + index * 90
        input_width = row["input_tokens"] / maximum * 560
        output_width = row["output_tokens"] / maximum * 560
        reasoning_width = row["reasoning_tokens"] / maximum * 560
        total_tokens = (
            row["input_tokens"] + row["output_tokens"] + row["reasoning_tokens"]
        )
        parts.extend(
            [
                f'<text x="70" y="{y+22}" font-family="Arial, sans-serif" font-size="17" fill="#172033">Arm {row["arm"]}</text>',
                f'<rect x="155" y="{y}" width="{input_width:.2f}" height="32" fill="#2563eb"/>',
                f'<rect x="{155+input_width:.2f}" y="{y}" width="{output_width:.2f}" height="32" fill="#93c5fd"/>',
                f'<rect x="{155+input_width+output_width:.2f}" y="{y}" width="{reasoning_width:.2f}" height="32" fill="#7c3aed"/>',
                f'<text x="{165+input_width+output_width+reasoning_width:.2f}" y="{y+22}" font-family="Arial, sans-serif" font-size="14" fill="#172033">{total_tokens:,}</text>',
            ]
        )
    parts.extend(
        [
            '<rect x="250" y="365" width="16" height="16" fill="#2563eb"/><text x="275" y="379" font-family="Arial, sans-serif" font-size="14" fill="#172033">input tokens</text>',
            '<rect x="430" y="365" width="16" height="16" fill="#93c5fd"/><text x="455" y="379" font-family="Arial, sans-serif" font-size="14" fill="#172033">output tokens</text>',
            '<rect x="610" y="365" width="16" height="16" fill="#7c3aed"/><text x="635" y="379" font-family="Arial, sans-serif" font-size="14" fill="#172033">reasoning tokens</text>',
        ]
    )
    return _svg_shell("Protocol-reported token use", "\n".join(parts))


def _summary(result: Mapping[str, Any]) -> bytes:
    h1, h2 = result["comparisons"]["H1"], result["comparisons"]["H2"]
    rates = result["valid_submission_rates"]
    text = f"""# Paper V1 confirmatory results

This document is generated from the content-hashed preregistered analysis. Do
not edit reported numbers by hand.

## Primary finding

Stateful iterative execution feedback did not improve the failure-adjusted
primary endpoint under the frozen budget. Arm B was lower than the budget-matched
best-of-N control (B − A = {h1['equal_stratum_mean_difference']:.3f}, 95% paired
stratified bootstrap CI [{h1['bootstrap_ci']['low']:.3f},
{h1['bootstrap_ci']['high']:.3f}], Holm-adjusted permutation p =
{h1['permutation']['holm_adjusted_p_value']:.6g}). Valid submissions occurred in
{rates['A']['valid_submissions']}/40 Arm A conditions ({rates['A']['rate']:.1%})
and {rates['B']['valid_submissions']}/40 Arm B conditions ({rates['B']['rate']:.1%}).

## Checklist increment

Adding checklist feedback did not show a clear incremental benefit over ordinary
iterative feedback (C − B = {h2['equal_stratum_mean_difference']:.3f}, 95% CI
[{h2['bootstrap_ci']['low']:.3f}, {h2['bootstrap_ci']['high']:.3f}],
Holm-adjusted permutation p = {h2['permutation']['holm_adjusted_p_value']:.6g}).
Arm C produced {rates['C']['valid_submissions']}/40 valid submissions
({rates['C']['rate']:.1%}).

## Evidence boundary

- The analysis includes all {result['completeness']['terminal_agent_outcomes']}
  terminal outcomes from {result['completeness']['planned_conditions']} frozen
  conditions; agent failures remain at dummy performance.
- Successful-only score summaries are descriptive and selection-biased.
- These results cover two internal LightLLM model endpoints, four tabular
  datasets, and five replicates per dataset-model-arm stratum; they are not a
  universal claim about all LLM agents or feedback designs.
- The post-collection reference-binding repair was made before any successful
  confirmatory statistic was emitted and remains disclosed as a limitation.

## Provenance

- Plan: `{result['plan_id']}` / `{result['plan_hash']}`
- Protocol: `{result['protocol_hash']}`
- Reconciliation: `{result['reconciliation_hash']}`
- Analysis result: `{result['result_hash']}`
"""
    return text.encode("utf-8")


def publication_files(result: Mapping[str, Any]) -> dict[str, bytes]:
    """Return the complete deterministic Paper V1 publication file set."""
    result = _validate_result(result)
    effect_rows = _primary_effect_rows(result)
    rate_rows = _valid_rate_rows(result)
    failure_rows = _failure_rows(result)
    resource_rows = _resource_rows(result)
    stratum_rows = _stratum_rows(result)
    hidden_rows = [
        {
            **row,
            "failure_adjusted_mean": _fmt(row["failure_adjusted_mean"]),
            "failure_adjusted_median": _fmt(row["failure_adjusted_median"]),
            "successful_only_mean": _fmt(row["successful_only_mean"]),
            "successful_only_median": _fmt(row["successful_only_median"]),
        }
        for row in result["hidden_score_summaries"]
    ]
    provenance = {
        "schema_version": "1.0",
        "source_evidence_class": result["evidence_class"],
        "plan_id": result["plan_id"],
        "plan_hash": result["plan_hash"],
        "protocol_hash": result["protocol_hash"],
        "analysis_reference_hash": result["analysis_reference_hash"],
        "analysis_config_hash": result["analysis_config_hash"],
        "reconciliation_hash": result["reconciliation_hash"],
        "analysis_result_hash": result["result_hash"],
    }
    files = {
        "primary-analysis.json": (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        "primary-effects.csv": _csv(effect_rows, list(effect_rows[0])),
        "valid-submission-rates.csv": _csv(rate_rows, list(rate_rows[0])),
        "failure-categories.csv": _csv(failure_rows, ("arm", "failure_category", "count")),
        "resource-usage.csv": _csv(resource_rows, list(resource_rows[0])),
        "stratum-effects.csv": _csv(stratum_rows, list(stratum_rows[0])),
        "hidden-score-summaries.csv": _csv(hidden_rows, HIDDEN_SUMMARY_FIELDS),
        "primary-effects.svg": _effect_svg(result),
        "valid-submission-rates.svg": _rate_svg(result),
        "failure-categories.svg": _failure_svg(result),
        "resource-usage.svg": _resource_svg(result),
        "SUMMARY.md": _summary(result),
        "provenance.json": (json.dumps(provenance, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    }
    return files


def publish_results(result: Mapping[str, Any], output_dir: str | Path) -> list[Path]:
    """Publish a deterministic set once; identical reruns are idempotent."""
    output = Path(output_dir)
    files = publication_files(result)
    for name, content in files.items():
        path = output / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite publication artifact: {path}")
    output.mkdir(parents=True, exist_ok=True)
    created = []
    for name, content in files.items():
        path = output / name
        if path.exists():
            continue
        with path.open("xb") as handle:
            handle.write(content)
        created.append(path)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Paper V1 tables and figures")
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    created = publish_results(result, args.output_dir)
    print(json.dumps({"created": [str(path) for path in created], "count": len(created)}, indent=2))


if __name__ == "__main__":
    main()

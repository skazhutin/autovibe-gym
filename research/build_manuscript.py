"""Build and verify the Paper V1 manuscript from frozen analysis artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from research.report_results import publication_files


RESULTS_TOKEN = "{{GENERATED_RESULTS}}"
REQUIRED_PACKAGE_FILES = (
    "research/build_manuscript.py",
    "research/audits/2026-08-16-confirmatory-results.md",
    "research/report_results.py",
    "research/run_artifacts.py",
    "research/publication/paper_v1/primary-analysis.json",
    "research/publication/paper_v1/primary-effects.csv",
    "research/publication/paper_v1/valid-submission-rates.csv",
    "research/publication/paper_v1/failure-categories.csv",
    "research/publication/paper_v1/resource-usage.csv",
    "research/publication/paper_v1/stratum-effects.csv",
    "research/publication/paper_v1/hidden-score-summaries.csv",
    "research/publication/paper_v1/primary-effects.svg",
    "research/publication/paper_v1/valid-submission-rates.svg",
    "research/publication/paper_v1/failure-categories.svg",
    "research/publication/paper_v1/resource-usage.svg",
    "research/publication/paper_v1/SUMMARY.md",
    "research/publication/paper_v1/provenance.json",
    "research/protocols/protocol_v1.yaml",
    "research/protocols/analysis_plan.md",
    "research/protocols/failure_policy.md",
    "research/protocols/freeze_manifest_v1.json",
    "research/protocols/freeze_record_v1.json",
    "paper/manuscript.template.md",
    "paper/references.bib",
    "paper/REPRODUCIBILITY.md",
    "paper/README.md",
    "tests/test_research_build_manuscript.py",
)


class ManuscriptError(ValueError):
    """Raised when the manuscript package cannot be reproduced exactly."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_text_sha256(data: bytes) -> str:
    """Hash UTF-8 package text with platform-independent LF newlines."""
    text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return _sha256(text.encode("utf-8"))


def _effect_row(result: Mapping[str, Any], hypothesis: str) -> str:
    item = result["comparisons"][hypothesis]
    return (
        f"| {hypothesis} | {item['treatment']} - {item['control']} | "
        f"{item['pair_count']} | {item['control_mean_fanu']:.3f} | "
        f"{item['treatment_mean_fanu']:.3f} | "
        f"{item['equal_stratum_mean_difference']:.3f} | "
        f"[{item['bootstrap_ci']['low']:.3f}, "
        f"{item['bootstrap_ci']['high']:.3f}] | "
        f"{item['permutation']['holm_adjusted_p_value']:.6f} |"
    )


def _results_section(result: Mapping[str, Any]) -> str:
    h1 = result["comparisons"]["H1"]
    h2 = result["comparisons"]["H2"]
    rates = result["valid_submission_rates"]
    failures = Counter(
        (row["arm"], row["failure_category"]) for row in result["rows"]
    )
    resources = {}
    for arm in ("A", "B", "C"):
        rows = [row for row in result["rows"] if row["arm"] == arm]
        resources[arm] = {
            "input": sum(row["input_tokens"] for row in rows),
            "output": sum(row["output_tokens"] for row in rows),
            "reasoning": sum(row["reasoning_tokens"] for row in rows),
            "calls": sum(row["logical_llm_calls"] for row in rows),
        }
        resources[arm]["total"] = (
            resources[arm]["input"]
            + resources[arm]["output"]
            + resources[arm]["reasoning"]
        )

    h1_strata = h1["dataset_model_summaries"]
    h2_strata = h2["dataset_model_summaries"]
    h1_positive = sum(row["mean_paired_difference"] > 0 for row in h1_strata)
    h1_zero = sum(row["mean_paired_difference"] == 0 for row in h1_strata)
    h1_negative = sum(row["mean_paired_difference"] < 0 for row in h1_strata)
    h2_positive = sum(row["mean_paired_difference"] > 0 for row in h2_strata)
    h2_zero = sum(row["mean_paired_difference"] == 0 for row in h2_strata)
    h2_negative = sum(row["mean_paired_difference"] < 0 for row in h2_strata)

    stratum_rows = []
    for h_id, items in (("H1", h1_strata), ("H2", h2_strata)):
        for item in items:
            stratum_rows.append(
                f"| {h_id} | {item['dataset_id']} | {item['model_id']} | "
                f"{item['mean_paired_difference']:.3f} |"
            )

    lines = [
        "<!-- BEGIN GENERATED RESULTS: do not edit by hand -->",
        "## 4 Results",
        "",
        "The completeness gate reconciled "
        f"{result['completeness']['terminal_agent_outcomes']} terminal agent outcomes "
        f"against {result['completeness']['planned_conditions']} planned conditions. "
        "There were no pending, running, replacement-required, or "
        "infrastructure-censored conditions in the final analysis population.",
        "",
        "### 4.1 Primary comparison: iterative feedback versus best-of-N",
        "",
        f"Arm A achieved a mean FANU of {h1['control_mean_fanu']:.3f}; Arm B "
        f"achieved {h1['treatment_mean_fanu']:.3f}. The equal-stratum paired "
        f"difference was {h1['equal_stratum_mean_difference']:.3f} (95% "
        f"bootstrap CI [{h1['bootstrap_ci']['low']:.3f}, "
        f"{h1['bootstrap_ci']['high']:.3f}]; two-sided permutation "
        f"p={h1['permutation']['p_value']:.6f}; Holm-adjusted "
        f"p={h1['permutation']['holm_adjusted_p_value']:.6f}). The interval "
        "excludes zero in the direction opposite to the motivating prediction: "
        "under this frozen budget and implementation, iterative feedback was "
        "worse than budget-matched independent attempts.",
        "",
        f"Valid submissions occurred in {rates['A']['valid_submissions']}/"
        f"{rates['A']['total_agent_outcomes']} Arm A conditions "
        f"({rates['A']['rate']:.1%}, 95% Wilson CI "
        f"[{rates['A']['confidence_interval']['low']:.1%}, "
        f"{rates['A']['confidence_interval']['high']:.1%}]) and "
        f"{rates['B']['valid_submissions']}/"
        f"{rates['B']['total_agent_outcomes']} Arm B conditions "
        f"({rates['B']['rate']:.1%}, 95% Wilson CI "
        f"[{rates['B']['confidence_interval']['low']:.1%}, "
        f"{rates['B']['confidence_interval']['high']:.1%}]). The exact paired "
        f"McNemar p-value was {h1['valid_submission_mcnemar']['p_value']:.8f}; "
        f"{h1['valid_submission_mcnemar']['control_valid_treatment_invalid']} "
        "pairs were valid only under A and "
        f"{h1['valid_submission_mcnemar']['treatment_valid_control_invalid']} "
        "only under B.",
        "",
        "### 4.2 Checklist increment",
        "",
        f"Arm C achieved a mean FANU of {h2['treatment_mean_fanu']:.3f}, "
        f"compared with {h2['control_mean_fanu']:.3f} for Arm B. The paired "
        f"difference was {h2['equal_stratum_mean_difference']:.3f} (95% CI "
        f"[{h2['bootstrap_ci']['low']:.3f}, "
        f"{h2['bootstrap_ci']['high']:.3f}]; Holm-adjusted permutation "
        f"p={h2['permutation']['holm_adjusted_p_value']:.6f}). Thus the study "
        "did not detect a clear incremental benefit from checklist feedback. "
        f"Arm C produced {rates['C']['valid_submissions']}/"
        f"{rates['C']['total_agent_outcomes']} valid submissions "
        f"({rates['C']['rate']:.1%}, 95% Wilson CI "
        f"[{rates['C']['confidence_interval']['low']:.1%}, "
        f"{rates['C']['confidence_interval']['high']:.1%}]); the B-versus-C "
        f"exact McNemar p-value was {h2['valid_submission_mcnemar']['p_value']:.6f}.",
        "",
        "### 4.3 Confirmatory effect table",
        "",
        "| Hypothesis | Contrast | Pairs | Control mean FANU | Treatment mean FANU | Difference | 95% CI | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _effect_row(result, "H1"),
        _effect_row(result, "H2"),
        "",
        "![Paired confirmatory effects](../research/publication/paper_v1/primary-effects.svg)",
        "",
        "### 4.4 Terminal outcomes and resource use",
        "",
        "The terminal category distribution differed sharply by arm. Arm A "
        f"had {failures[('A', 'success')]} successes and "
        f"{failures[('A', 'invalid_submission')]} invalid submissions. Arm B "
        f"had {failures[('B', 'success')]} successes and "
        f"{failures[('B', 'budget_exhausted')]} budget-exhausted outcomes; Arm C "
        f"had {failures[('C', 'success')]} successes and "
        f"{failures[('C', 'budget_exhausted')]} budget-exhausted outcomes. "
        "Budget exhaustion is therefore the dominant observed terminal mechanism "
        "for the iterative arms, although the present experiment does not isolate "
        "which component of the iterative policy produced it.",
        "",
        "| Arm | Valid / 40 | Input tokens | Output tokens | Reasoning tokens | Total protocol tokens | LLM calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("A", "B", "C"):
        item = resources[arm]
        lines.append(
            f"| {arm} | {rates[arm]['valid_submissions']} | {item['input']:,} | "
            f"{item['output']:,} | {item['reasoning']:,} | {item['total']:,} | "
            f"{item['calls']:,} |"
        )
    lines.extend(
        [
            "",
            f"Arms B and C consumed {resources['B']['total'] / resources['A']['total']:.2f}x "
            f"and {resources['C']['total'] / resources['A']['total']:.2f}x the "
            "protocol-reported tokens of Arm A, respectively. Token volume is not "
            "converted to money because traffic used an internal endpoint and "
            "operator-side billing was not independently observable.",
            "",
            "The preregistered resource plan also called for cached-token, code- "
            "and tool-execution, wall-clock, CPU-time, provider-retry, monetary- "
            "cost, and cost-quality-frontier reporting. Those fields were not "
            "emitted by the frozen primary-analysis result schema. We record this "
            "as a reporting deviation rather than reconstructing a post-outcome "
            "secondary analysis here; consequently, the resource evidence in this "
            "manuscript is limited to input/output/reasoning tokens and logical "
            "model calls.",
            "",
            "![Terminal outcomes by arm](../research/publication/paper_v1/failure-categories.svg)",
            "",
            "![Protocol-reported token use](../research/publication/paper_v1/resource-usage.svg)",
            "",
            "### 4.5 Heterogeneity",
            "",
            f"For H1, {h1_positive} of eight dataset-model strata had a positive "
            f"mean difference, {h1_zero} was exactly zero, and {h1_negative} were "
            f"negative. For H2, the corresponding counts were {h2_positive} "
            f"positive, {h2_zero} zero, and {h2_negative} negative. These small "
            "stratum samples are descriptive rather than separately powered "
            "hypothesis tests.",
            "",
            "| Hypothesis | Dataset | Model endpoint | Mean paired FANU difference |",
            "|---|---|---|---:|",
            *stratum_rows,
            "",
            "Checklist coverage is intentionally absent from the results: the "
            "preregistered two-annotator detector validation has not been "
            "completed, so detector outputs are not treated as validated evidence.",
            "<!-- END GENERATED RESULTS -->",
        ]
    )
    return "\n".join(lines)


def validate_publication_directory(
    result: Mapping[str, Any], publication_dir: Path
) -> None:
    """Require every committed derived artifact to match canonical rendering."""
    expected_files = publication_files(result)
    missing = []
    changed = []
    for name, expected in expected_files.items():
        path = publication_dir / name
        if not path.is_file():
            missing.append(name)
        elif canonical_text_sha256(path.read_bytes()) != canonical_text_sha256(
            expected
        ):
            changed.append(name)
    if missing or changed:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if changed:
            details.append("changed=" + ",".join(sorted(changed)))
        raise ManuscriptError(
            "publication directory does not match canonical analysis rendering: "
            + "; ".join(details)
        )


def build_manuscript(root: Path) -> tuple[bytes, bytes]:
    analysis_path = root / "research/publication/paper_v1/primary-analysis.json"
    if not analysis_path.is_file():
        raise ManuscriptError(f"required analysis file is missing: {analysis_path}")
    result = json.loads(analysis_path.read_text(encoding="utf-8"))
    validate_publication_directory(result, analysis_path.parent)

    template_path = root / "paper/manuscript.template.md"
    template = template_path.read_text(encoding="utf-8")
    if template.count(RESULTS_TOKEN) != 1:
        raise ManuscriptError("manuscript template must contain one results token")
    manuscript = template.replace(RESULTS_TOKEN, _results_section(result))
    manuscript_bytes = manuscript.encode("utf-8")

    file_hashes = {}
    for relative in REQUIRED_PACKAGE_FILES:
        path = root / relative
        if not path.is_file():
            raise ManuscriptError(f"required package file is missing: {relative}")
        file_hashes[relative] = canonical_text_sha256(path.read_bytes())
    file_hashes["paper/manuscript.md"] = canonical_text_sha256(manuscript_bytes)
    manifest = {
        "schema_version": "1.0",
        "package": "autovibe-gym-paper-v1-working-manuscript",
        "publication_status": "internal_working_manuscript_not_released",
        "file_hash_policy": "sha256_utf8_text_normalized_to_lf",
        "analysis_result_hash": result["result_hash"],
        "plan_id": result["plan_id"],
        "plan_hash": result["plan_hash"],
        "protocol_hash": result["protocol_hash"],
        "reconciliation_hash": result["reconciliation_hash"],
        "files": dict(sorted(file_hashes.items())),
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    return manuscript_bytes, manifest_bytes


def write_or_check(root: Path, *, check: bool) -> None:
    manuscript, manifest = build_manuscript(root)
    expected = {
        root / "paper/manuscript.md": manuscript,
        root / "paper/artifact-manifest.json": manifest,
    }
    if check:
        stale = [
            str(path)
            for path, content in expected.items()
            if not path.is_file()
            or canonical_text_sha256(path.read_bytes())
            != canonical_text_sha256(content)
        ]
        if stale:
            raise ManuscriptError("paper package is stale: " + ", ".join(stale))
        return
    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Paper V1 manuscript")
    parser.add_argument("--root", default=".")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_or_check(Path(args.root).resolve(), check=args.check)
    print(json.dumps({"status": "current" if args.check else "built"}))


if __name__ == "__main__":
    main()

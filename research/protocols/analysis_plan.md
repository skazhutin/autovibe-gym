# Analysis plan

This plan is authoritative together with `protocol_v1.yaml`. If prose and YAML
diverge, analysis stops until the protocol is amended and revalidated before
confirmatory outcomes are inspected.

## Analysis populations

1. **Planned conditions:** the complete precomputed A/B/C matrix.
2. **Agent-outcome population:** one terminal agent outcome per planned condition,
   after replacing predeclared provider/infrastructure failures.
3. **Successful submissions:** the subset with a valid one-shot hidden evaluation.
4. **Infrastructure attempts:** every provider/orchestrator/sandbox attempt,
   retained for reliability and cost reporting even when censored.

Primary FANU uses the agent-outcome population, not successful submissions only.

## Data checks before analysis

- Verify protocol, config, Git, dataset, split, prompt, model, budget, and
  execution-policy hashes.
- Reject dirty-worktree confirmatory runs.
- Verify exactly one terminal agent outcome per planned condition.
- Verify every rerun has a unique `run_id`, `rerun_of`, and reason.
- Reconcile usage totals from the per-call ledger and execution totals from the
  execution ledger; do not trust independently hand-edited totals.
- Confirm no agent-facing artifact contains hidden rows, labels, score, evaluator
  diagnostics, private paths, or secrets.
- Confirm hidden evaluation count is at most one per terminal agent outcome.
- Produce a completeness table before computing arm effects.

Any violation stops the primary pipeline and produces a machine-readable error;
it is not repaired by dropping inconvenient rows.

## Primary endpoint: FANU

For each frozen dataset, record metric direction, `S_dummy`, and `S_reference`
before confirmatory runs.

Higher-is-better metrics:

```text
FANU = (S_agent - S_dummy) / (S_reference - S_dummy)
```

Lower-is-better metrics:

```text
FANU = (S_dummy - S_agent) / (S_dummy - S_reference)
```

Rules:

- agent-caused failure or invalid submission: `S_agent = S_dummy`;
- infrastructure-censored attempt: no FANU; queue replacement under the same
  condition and retain the failed attempt separately;
- successful score: use the only hidden-test score for that terminal outcome;
- do not clip FANU; values below 0 and above 1 are valid;
- do not impute an unobserved hidden score from validation score.

If `S_reference == S_dummy`, the dataset is invalid for FANU and must be replaced
or the endpoint amended before freeze.

## Primary and key secondary comparisons

Primary H1 comparison: paired `B - A` FANU.

Key secondary H2 comparison: paired `C - B` FANU.

Blocks are `dataset_id × model_id × replicate_index`. Aggregate estimates give
equal weight to each dataset-model stratum. Within each stratum, replicate pairs
are equally weighted.

For each comparison report:

- arm means/medians and paired mean/median differences;
- 95% paired stratified percentile bootstrap CI with 10,000 resamples, seed
  `20260812`;
- two-sided paired permutation test, exact when feasible and otherwise 100,000
  Monte Carlo draws with seed `20260812`; exact means at most 20 non-zero
  paired differences, and Monte Carlo p-values use the add-one correction;
- Holm-adjusted p-values across the H1 and H2 FANU tests;
- raw paired values and per-dataset/per-model summaries.

Effect estimates and intervals are primary; p-values are supporting evidence.
Bootstrap resampling is within each dataset-model stratum, with the final
statistic equally averaging stratum means. Permutation sign flips use the same
equal-stratum statistic, including if strata contain unequal observed pair
counts.

## Valid submission and failure outcomes

- Compare paired valid/failed status with exact McNemar testing.
- Report arm-specific rates with 95% Wilson score confidence intervals.
- Show every failure category by arm, dataset, and model.
- Report hidden-test score twice: over all agent outcomes using the declared
  failure adjustment, and conditionally among successful submissions. The latter
  is descriptive and may be selection-biased.
- Define error recovery as a prior recorded agent error followed by a valid
  submission within the same episode; report numerator and denominator.

## Resource and Pareto analysis

Report raw input, output, reasoning, and cached tokens separately, plus the
protocol token total. Also report LLM calls, code executions, tool calls,
wall-clock, CPU time, provider retries, and monetary cost.

Construct cost-quality frontiers without collapsing token volume and currency
into one measure. Plain single-shot, if retained, appears only as a reference
point and not as the H1 budget-matched control.

## Checklist validation

Sample 20–25% of trajectories using a precomputed stratified sample over arm,
dataset, and model. Two people independently annotate all 12 mandatory items from
code and logs without seeing hidden scores.

Report per-item precision, recall, and F1 for the automatic detector and Cohen's
kappa or Krippendorff's alpha for human agreement. Preserve raw annotations,
disagreements, adjudication decisions, and the annotation-guide version.

If detector accuracy is inadequate, retain checklist coverage only as an
explicitly noisy process proxy and do not use it as evidence of solution quality.

## Missing data, exclusions, and sensitivity analyses

- Agent failures are not missing and remain in FANU at dummy performance.
- Provider/infrastructure failures are censored and replaced; report the original
  attempts and replacement count.
- No complete-case-only primary analysis.
- Every exclusion requires a predeclared rule and an exclusion-log entry written
  without using arm outcomes.
- Sensitivity analyses: successful-only hidden score, no-replacement view of
  infrastructure attempts, median paired difference, and results by dataset/model.
- Mixed-effects models are exploratory only.

## Result-blind freeze and provenance

The confirmatory configuration, condition matrix, analysis code, dataset hashes,
reference scores, and exclusions must be committed and tagged before outcomes are
inspected. Tables and figures must be generated from immutable manifests; manual
copying of reported numbers is prohibited.

The immutable plan records canonical hashes of both the complete protocol and
the frozen FANU reference payload. The primary pipeline recomputes both and
stops on mismatch, then propagates both hashes into its result.

The primary analysis output is content-hashed and published with no-overwrite
semantics. An identical rerun is idempotent; a different output at the same path
is an error rather than an implicit replacement.

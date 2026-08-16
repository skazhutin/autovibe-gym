# Paper V1 Confirmatory Results Audit

Date: 2026-08-16

## Evidence set

- Frozen plan: `plan_793d85d310427d848a09302c`
- Plan hash: `793d85d310427d848a09302c3c4583870fd3fafb6a8a0590d95e61384edf310f`
- Protocol hash: `de690a4e975781c516b25d2016416bf44e5d416ee61a00d305eae39331ce49c5`
- Reconciliation hash: `805c6dfd3da10a9ae33ea80276d3643df2b2ec17fa64ca83cb781ddbee844e16`
- Primary-analysis result hash: `22ff7ac1ffeb7ce116c384ea32ab23f82984c01aa7b16f19993b8f8f1db3c58a`
- Primary-analysis file SHA-256: `6cbe78eea2de19e85a350cb05521057214b47e44f9282a87cb6658a50b5c2948`

The immutable series reconciled all 120 planned conditions: 40 per arm, with
zero pending, running, replacement-required, or infrastructure-censored
conditions. The primary analysis used all 120 terminal agent outcomes. Raw runs
remain outside Git; the committed publication copy is byte-identical to the
external content-hashed analysis file.

## Confirmatory findings

The primary hypothesis was not supported in the predicted direction. Relative
to budget-matched best-of-N single-shot (Arm A), iterative execution feedback
(Arm B) reduced failure-adjusted normalized utility:

- equal-stratum mean `B - A`: `-0.731124`;
- 95% paired stratified bootstrap CI: `[-0.848360, -0.608697]`;
- Holm-adjusted permutation p-value: `0.000020`;
- valid submissions: A `30/40` (`75.0%`), B `6/40` (`15.0%`);
- exact paired McNemar p-value: `0.000001` after six-decimal rendering.

The checklist increment was not distinguishable from zero at the registered
precision:

- equal-stratum mean `C - B`: `-0.048985`;
- 95% paired stratified bootstrap CI: `[-0.211189, 0.107149]`;
- Holm-adjusted permutation p-value: `0.570312`;
- valid submissions: C `3/40` (`7.5%`), versus B `6/40` (`15.0%`);
- exact paired McNemar p-value: `0.453125`.

Arm A produced 30 successes and 10 invalid submissions. Arm B produced six
successes and 34 budget-exhausted outcomes; Arm C produced three successes and
37 budget-exhausted outcomes. Protocol-reported input plus output tokens were
`1,545,371` for A, `4,295,775` for B, and `4,467,438` for C. Token volume and
quality are reported separately; no monetary value is inferred from internal
LightLLM traffic.

## Verification

- The first analysis invocation failed closed before computing statistics due
  to the reference-binding implementation defect recorded in the amendment.
- PR #72 repaired only that binding and passed local `384 passed` plus GitHub
  `Python tests` before merge.
- The successful analysis ran once to a fresh `analysis-v2` path; a different
  payload cannot overwrite it.
- The result validates against `analysis_result.schema.json`, its canonical
  result hash, and the frozen plan/protocol/reference hashes.
- The reporting generator is deterministic and idempotent. An identical rerun
  creates zero files, while a differing existing artifact is rejected.
- A secret-pattern scan of the publication directory returned no matches.
- Focused reporting/analysis/protocol tests: `21 passed`.
- Full repository suite: `385 passed, 2 skipped`, with one existing Jupyter path
  deprecation warning.

## Deviations and limitations

1. The post-collection reference-binding repair occurred before any successful
   confirmatory statistic was emitted. It did not alter an estimand, endpoint,
   failure rule, resampling rule, or run.
2. GitHub rebase-merge rewrote the reviewed PR #72 commit into `4ab8804` with
   author `JapanDino <klim.i.rumyantsev@gmail.com>` but committer display name
   `J D <klim.i.rumyantsev@gmail.com>`. The authenticated merge account was
   `JapanDino`, and the source PR commit `6698a58` has exact author/committer
   identity. The owner instructed work to continue; published `main` history was
   not force-rewritten. Future publication commits must preserve exact identity.
3. Human checklist-detector annotation has not been completed. No checklist
   coverage claim is treated as validated evidence in PR 5b or the manuscript.
4. The endpoint set contains two internal service model IDs without immutable
   weight revisions. Request-level provenance and the service snapshot are the
   available model identity boundary.
5. The analysis is limited to four tabular datasets, two model endpoints, and
   five replicates per dataset-model-arm stratum. It is evidence against a broad
   benefit in this frozen setup, not proof that iterative feedback never helps.
6. The frozen primary-analysis result schema emitted input, output, and reasoning
   tokens plus logical model calls, but omitted preregistered cached-token,
   code/tool-execution, wall-clock, CPU-time, provider-retry, monetary-cost, and
   cost-quality-frontier fields. PR 6 records this as an incomplete resource-
   reporting deviation rather than constructing an unregistered post-outcome
   secondary analysis.
7. The frozen result/reporting schema also omitted registered
   `validation_to_test_gap`, `error_recovery_rate`, `clean_replay_success`, and
   failure-category tables by arm/dataset/model. Aggregate arm failure counts
   are reported and row-level dataset/model/category fields remain available,
   but the omitted reports are not reconstructed post outcome in PR 6.

## Publication boundary

This audit supports PR 5b tables, figures, and manuscript drafting. It does not
authorize a journal submission, public release, or universal product claim.

# Paper V1 Analysis Reference Binding Repair

Date: 2026-08-16

## Discovery point

The confirmatory series had reached the preregistered completeness gate:
120 terminal conditions, zero pending, zero running, and zero
replacement-required conditions. The first invocation of the primary analysis
then stopped before computing or emitting any statistics with
`references.plan_id does not match the immutable plan`.

No confirmatory score, arm summary, effect estimate, confidence interval, or
hypothesis-test result was inspected before this repair.

## Cause

The frozen FANU reference file is created before the confirmatory plan and its
canonical payload contains `schema_version`, `reference_pipeline_id`, and the
ordered dataset references. The plan then binds that payload through
`analysis_reference_hash`.

The analysis implementation and its synthetic fixture instead required the
reference file to contain the later-created `plan_id` and `plan_hash`, which
would make the two immutable objects depend on each other. It also recomputed
the reference hash from a different payload that omitted
`reference_pipeline_id` and reordered datasets. The repository's actual frozen
reference file therefore could not pass the preregistered analysis gate.

## Repair

The analysis gate now:

1. requires the exact fields written by the frozen reference generator;
2. recomputes the reference hash from that exact canonical payload;
3. requires the recomputed hash to equal both `reference_hash` in the frozen
   file and `analysis_reference_hash` in the immutable plan.

The JSON schema and synthetic regression fixtures are aligned with the same
one-way binding.

## Invariants

This repair does not change the frozen plan, condition order, datasets, model
identities, prompts, budgets, failure policy, scores, FANU formula, estimands,
resampling seeds, confidence intervals, hypothesis tests, multiplicity
correction, or reporting rules. It does not replace or rerun any confirmatory
condition.

The repair is an implementation correction discovered after data collection
but before the first successful confirmatory analysis. It must be reported in
the audit trail and manuscript limitations.

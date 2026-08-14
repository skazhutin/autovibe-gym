# AutoVibe Gym Paper V1 research package

This directory defines the research contract for the working paper:

> **When Does Feedback Pay Off? A Budget-Matched, Failure-Aware Evaluation of
> LLM Agents for Tabular Machine Learning**

The package is deliberately protocol-first. Existing experiment outputs remain
pilot evidence and must not be merged with the future confirmatory series.

## Current state

- Protocol status: `draft_unfrozen`.
- Phase 0 code audit: complete at Git commit `1504cc0` (`origin/main` on
  2026-08-12).
- Confirmatory runs: not authorized and not technically ready.
- Runner causal behavior: unchanged; opt-in Paper V1 artifacts do not alter
  prompts, budgets, submit behavior, or provider retry decisions.
- Manifest/ledger and failure-classification infrastructure: implemented in
  PR 2; fair global budgets and the confirmatory planner remain later PRs.

The protocol cannot be frozen until every freeze-blocking TODO in
[`protocol_v1.yaml`](protocols/protocol_v1.yaml) is resolved by a human owner.

## Arm mapping

The paper uses stable analysis arm IDs even though current product labels differ:

| Arm | Analysis concept | Current product mode | Role |
|---|---|---|---|
| A | budget-matched best-of-N single-shot | `repeated_single_shot` | confirmatory control |
| B | iterative execution feedback without checklist | `iterative_no_checklist` | confirmatory treatment |
| C | iterative execution feedback with checklist | `gym_with_checklist` | confirmatory checklist ablation |
| D | fixed workflow transitions | `fixed_transitions` | exploratory only |

Plain `single_shot` may be retained as a cost/Pareto reference after owner
approval, but it is not a budget-matched causal control.

## Research invariants

Confirmatory execution must satisfy all of the following:

1. A/B/C share one global token, call, execution, tool, wall-clock, CPU, and RAM
   budget contract.
2. The evaluator never fits, repairs, or otherwise improves an agent submission.
3. Validation may guide selection; hidden-test score or diagnostics may not.
4. Hidden evaluation is attempted at most once per agent outcome and produces no
   feedback that can affect that outcome.
5. Provider and infrastructure failures are censored and replaced under the
   declared policy; agent failures remain outcomes.
6. Every planned condition exists before confirmatory results are inspected.
7. Protocol/config/code/dataset/prompt hashes and immutable run artifacts connect
   every reported number to its provenance.
8. No claim that one arm is better is made without matched budgets, repeats,
   effect sizes, and confidence intervals.

## Files

- [`protocols/protocol_v1.yaml`](protocols/protocol_v1.yaml): canonical
  machine-readable protocol.
- [`protocols/hypotheses.md`](protocols/hypotheses.md): estimands and claim rules.
- [`protocols/analysis_plan.md`](protocols/analysis_plan.md): preprocessing,
  missing-data, and statistical analysis rules.
- [`protocols/failure_policy.md`](protocols/failure_policy.md): failure taxonomy,
  retry, censoring, and replacement rules.
- [`protocols/selection_requirements.md`](protocols/selection_requirements.md):
  dataset and model eligibility gates.
- [`audits/2026-08-12-phase0.md`](audits/2026-08-12-phase0.md): evidence-backed
  map of the current implementation and gaps.
- [`run_artifacts.py`](run_artifacts.py): stable condition IDs, unique run IDs,
  atomic manifests, append-only usage/execution ledgers, redaction, and terminal
  failure classification.
- [`schemas/run_manifest.schema.json`](schemas/run_manifest.schema.json):
  versioned manifest interchange schema.

## Opt-in run artifacts

Existing product runs remain unchanged unless both research flags are present:

```powershell
python -m experiments.run_gym `
  --dataset-dir datasets/demo/prepared `
  --model <registry-model-id> `
  --research-run-dir outputs/paper-v1-pilot `
  --research-experiment-id paper-v1-availability-pilot
```

Each attempt creates a new `run_<uuid>/` directory. An explicit duplicate
`--research-run-id` fails instead of overwriting prior evidence. Technical
reruns additionally require `--research-rerun-of` and
`--research-rerun-reason`. The generated artifacts are:

- `run_manifest.json` — atomically replaced state document with condition,
  code/config/data/prompt hashes, terminal category, and ledger totals;
- `usage_ledger.jsonl` — append-only logical calls plus provider request
  attempts and retry indices where the provider adapter exposes them;
- `execution_ledger.jsonl` — append-only baseline executions or public
  notebook events, with private evaluator fields removed;
- `manifest_events.jsonl` — append-only lifecycle audit.

The current provider retry limits and backoff are only observed. PR 2 does not
change them or enforce the future common global budget. The default model
version is recorded honestly as `unversioned`; such runs are pilot-only and
cannot satisfy the later freeze gate.

## Offline validation

```powershell
python -m research.validate_protocol research/protocols/protocol_v1.yaml
python -m pytest tests/test_research_protocol.py -q
python -m pytest tests/test_research_run_artifacts.py tests/test_llm.py -q
```

The validator checks structural consistency only. A passing validation does not
freeze the protocol, approve unresolved human decisions, or make the current
runners confirmatory-ready.

## Phase boundaries

- Detailed Draft/Ready/Merge/freeze gates are defined in
  [`docs/RESEARCH_PR_POLICY.md`](../docs/RESEARCH_PR_POLICY.md).
- **PR 1:** protocol and audit only.
- **PR 2:** `RunManifest`, stable condition IDs, usage/execution ledgers, and
  failure classification without changing causal mode behavior.
- **PR 3:** fair global budget, common submission validator, research-mode
  autofit removal, and one-shot hidden evaluation.
- **PR 4:** precomputed matrix, blocked randomization, resume/idempotency, and
  replacement queue.
- **PR 5a:** preregister the FANU/completeness/paired-analysis implementation on
  synthetic manifests before protocol freeze.
- **Freeze + confirmatory runs:** only after PR 1–5a, the result-blind pilot,
  resolved human decisions, a clean acceptance gate, and an immutable tag.
- **PR 5b:** populate tables/figures from the immutable confirmatory manifest set.
- **PR 6:** paper and reproducibility package.

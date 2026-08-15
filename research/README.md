# AutoVibe Gym Paper V1 research package

This directory defines the research contract for the working paper:

> **When Does Feedback Pay Off? A Budget-Matched, Failure-Aware Evaluation of
> LLM Agents for Tabular Machine Learning**

The package is deliberately protocol-first. Existing experiment outputs remain
pilot evidence and must not be merged with the future confirmatory series.

## Current state

- Protocol status: result-blind freeze amendment v2 is awaiting review and tag
  `paper-v1-experiment-freeze-v2`; it supersedes the immutable original tag.
- Phase 0 code audit: complete at Git commit `1504cc0` (`origin/main` on
  2026-08-12).
- Confirmatory runs: authorized but not started; the fail-closed launcher must
  pass review before the first condition is executed.
- Runner causal behavior: opt-in research mode now uses the shared Paper V1
  global budget, common no-autofit validator, and one-shot hidden evaluation;
  product defaults remain compatible.
- Manifest/ledger and failure-classification infrastructure: merged in PR 2.
- Fair global budgets and research submission gates: merged in PR 3.
- Confirmatory planner: merged in PR 4; the frozen plan contains exactly 120
  pending conditions (40 randomized A/B/C blocks).
- Primary analysis pipeline: preregistered in PR 5a and tested only on synthetic
  manifests; it does not contain or imply a confirmatory result.

The original contract and exact plan were merged in PR #69 and anchored by tag
`paper-v1-experiment-freeze`. The first launch stopped before manifest creation
or provider access because integer `1800` and runner-emitted float `1800.0`
produced different canonical hashes. The v2 amendment changes only that numeric
serialization, regenerates the complete plan before outcomes, and retains the
original tag and failed-launch audit trail. See
[`protocols/amendments/2026-08-15-freeze-v2.md`](protocols/amendments/2026-08-15-freeze-v2.md).

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
- [`planner.py`](planner.py): deterministic blocked condition planning,
  immutable plan writing, manifest reconciliation, resume checks, and the
  infrastructure-replacement queue.
- [`schemas/confirmatory_plan_config.schema.json`](schemas/confirmatory_plan_config.schema.json):
  exact result-blind planner input contract.
- [`schemas/confirmatory_plan.schema.json`](schemas/confirmatory_plan.schema.json):
  immutable expanded-plan interchange schema.
- [`analysis.py`](analysis.py): fail-closed completeness, FANU, paired H1/H2,
  stratified bootstrap, permutation, McNemar, Holm, and immutable result output.
- [`schemas/analysis_references.schema.json`](schemas/analysis_references.schema.json):
  frozen dataset metric-direction/dummy/reference input contract.
- [`schemas/analysis_result.schema.json`](schemas/analysis_result.schema.json):
  content-hashed primary-analysis result contract.

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

Provider retry limits and backoff remain observed rather than altered by the
artifact layer. PR 3 adds the opt-in common global episode budget around those
calls. A model version recorded as `unversioned` remains pilot-only and cannot
satisfy the later freeze or confirmatory-plan gate.

## Confirmatory planner

The planner does not infer unresolved model, dataset, budget, prompt, reference,
or execution decisions. Its input must contain exact Git/config/dataset/split/
model identities and SHA-256 hashes for the budget, frozen FANU-reference,
decoding, execution, and prompt contracts. Values such as `TODO`, `TBD`,
`unversioned`, or `unknown` are rejected.

After those owner decisions are resolved, build the complete plan without
executing any episode:

```powershell
python -m research.planner build `
  --config path/to/frozen-plan-input.yaml `
  --protocol research/protocols/protocol_v1.yaml `
  --output path/to/confirmatory-plan.json
```

The generated plan contains every A/B/C condition before outcomes are visible,
uses a portable SHA-256-seeded permutation within each
dataset/model/replicate block, and hashes the complete canonical plan. Repeating
the same build against the same output is a no-op; a different plan is never
allowed to overwrite it.

Resume and replacement status is derived only from append-only run manifests:

```powershell
python -m research.planner status `
  --plan path/to/confirmatory-plan.json `
  --runs path/to/immutable-run-root `
  --output path/to/reconciliation.json
```

Agent failures remain terminal outcomes. Only declared infrastructure/provider
failures enter the same-condition replacement queue, with a new run ID,
`rerun_of`, and reason. Unknown conditions, duplicate run IDs, multiple terminal
outcomes, broken replacement chains, or condition drift fail reconciliation.
This tooling is confirmatory infrastructure only: it neither authorizes nor
launches pilot/confirmatory API runs.

## Confirmatory launcher

The launcher schedules the next condition from the frozen plan and defaults to
one condition per invocation. It fails before provider access unless all of the
following still match: annotated freeze tag and complete JSON contract, clean
detached execution worktree at the bound commit, dataset hashes, exact internal
OpenAI-compatible model records, Docker image availability, and manifest/event
history.

```powershell
python -m research.confirmatory_launcher `
  --execution-repo path/to/detached-frozen-worktree `
  --datasets-root path/to/frozen-datasets `
  --runs-root path/to/external-results/runs `
  --models-config path/to/private-models.json `
  --sandbox-image autovibe-gym-sandbox:paper-v1 `
  --dry-run
```

Remove `--dry-run` only after review. `--max-conditions N` permits a bounded
sequence; the default remains one. The private registry stays outside Git and
supplies the internal API credential at runtime. There is no paid-provider
fallback. Stdout/stderr, workspaces, MLflow SQLite state, run manifests, and the
append-only launcher lifecycle ledger all live beside the external runs root.

The Docker image was built after the protocol tag, so its digest is not a field
of the preregistered condition payload. The launcher records the first selected
digest as common infrastructure provenance and rejects any digest change for
the remainder of the series. A missing/unfinished lifecycle event, running
manifest, contract drift, or infrastructure replacement stops automation for
human review; no result-based rerun decision is made.

## Preregistered primary analysis

After a frozen plan exists and its complete immutable run set has been
reconciled, run the primary analysis without manually copying outcomes:

```powershell
python -m research.analysis `
  --plan path/to/confirmatory-plan.json `
  --runs path/to/immutable-run-root `
  --references path/to/frozen-fanu-references.json `
  --protocol research/protocols/protocol_v1.yaml `
  --output path/to/primary-analysis.json
```

The pipeline refuses incomplete matrices, condition drift, invalid replacement
chains, mismatched plan/reference identities, zero FANU denominators, successful
runs without exactly one hidden evaluation, dirty or unknown worktree
provenance, ledger/manifest total drift, and failed runs marked as valid. It
recomputes resource totals from immutable usage/execution ledgers, keeps agent
failures at dummy performance, and rejects infrastructure attempts as terminal
outcomes. H1 (`B-A`) and H2 (`C-B`) are paired by
dataset/model/replicate, aggregate dataset-model strata equally, and apply the
protocol bootstrap, permutation, exact McNemar, and Holm procedures.
Arm-specific valid-submission rates use the preregistered 95% Wilson score
interval; the paired rate comparison remains exact McNemar.
Per-arm/dataset/model hidden-score summaries include both failure-adjusted all-
outcome statistics and successful-only sensitivity statistics, with the latter
explicitly marked as selection-biased.

The immutable plan carries both the canonical protocol hash and the canonical
FANU-reference hash. Analysis recomputes both hashes, refuses a mismatch, and
propagates them into the result so seeds, resample counts, directions, dummy
scores, and reference scores cannot be changed after outcomes are visible.

The output path is concurrent-safe and cannot be overwritten. This PR validates
the pipeline only with synthetic manifests; confirmatory values remain
unavailable until the pilot, freeze, immutable tag, and execution gates pass.

## Offline validation

```powershell
python -m research.validate_protocol research/protocols/protocol_v1.yaml
python -m pytest tests/test_research_protocol.py -q
python -m pytest tests/test_research_run_artifacts.py tests/test_llm.py -q
python -m pytest tests/test_research_planner.py -q
python -m pytest tests/test_research_analysis.py -q
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

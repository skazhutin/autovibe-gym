# AutoVibe Gym - Live Status

**Last updated:** 2026-08-15 (Paper V1 freeze package and exact 120-condition plan are complete; confirmatory execution has not started)
**Phase:** Paper V1 freeze review before immutable tag and confirmatory execution.

---

## Current Sprint Goal

Review and merge an auditable, result-blind Paper V1 freeze with its complete
precomputed plan, then create and verify the immutable experiment tag before
executing any confirmatory condition.

## Paper V1 Research Protocol

| Item | Status | Notes |
|---|---|---|
| Phase 0 code audit | Done | Audited freshly fetched `origin/main` commit `1504cc0`; mapped five modes, per-call token semantics, host autofit, step/tool counters, artifact paths, hidden-score privacy, hidden retry loop, and current tests |
| `research/protocol-v1` scaffold | Done | Machine-readable protocol, hypotheses, analysis plan, failure policy, selection gates, validator, and offline tests |
| Protocol freeze | Ready for review | All D01-D09 decisions are recorded; two internal model IDs, four leakage-reviewed dataset snapshots, 256k/12-call budget, zero-ruble no-fallback policy, storage choice, annotator nominees, manifest `91e4a9...d7ac`, and FANU references `e86777...0570` are bound. Freeze becomes operative only after merge and tag `paper-v1-experiment-freeze` |
| Confirmatory execution | Planned, not started | The immutable result-blind plan contains exactly 120 pending conditions (40 A/B/C blocks) and binds execution commit `b4e3da2`; nine completed availability/budget pilot episodes remain explicitly non-confirmatory and no confirmatory outcome has been generated or inspected |
| Candidate prediction isolation | Implemented, under review | Confirmatory Docker runs explicitly bind agent execution and candidate prediction to Docker. Readiness and hidden evaluation run in a separate ephemeral, network-none, read-only/cap-dropped evaluator; its result crosses to the host only as a validated JSON scalar vector, never executable pickle output |
| Auditable run artifacts (PR 2) | Merged | PR 2 was rebased onto merged PR 1, reverified (`289 passed, 2 skipped` locally plus required GitHub `Python tests` success), and squash-merged as `67069a3` |
| Causal runner behavior changes (PR 3) | Merged as PR #65 | Research-mode A/B/C runners share one pre-call-enforced episode budget, common no-autofit submission validation, and at most one hidden evaluation. Default product behavior remains compatible. After review fixes, `307 passed, 2 skipped` locally and both GitHub checks passed. Fast-forward merge preserved all three commits with author/committer `JapanDino`; `main` advanced to `385a09f` |
| Confirmatory experiment planner (PR 4) | Merged as PR #66 | Deterministic immutable A/B/C matrix expansion, portable blocked randomization, plan/config hashes, concurrent-safe no-overwrite writes, manifest-based resume, duplicate/condition-drift/category rejection, and same-condition infrastructure replacement queue. Review-fix full suite: `338 passed, 2 skipped`; both GitHub checks passed. Fast-forward merge preserved JapanDino authorship and advanced `main` to `1b12017` |
| Preregistered primary analysis (PR 5a) | Merged as PR #67 | Synthetic-only FANU, fail-closed completeness, paired B-A/C-B analysis, equal-stratum bootstrap/permutation, exact McNemar, 95% Wilson rate intervals, all-outcome/successful-only score summaries, ledger reconciliation, Holm correction, raw pairs/stratum summaries, content hashes, schemas, and concurrent-safe no-overwrite output. Final focused suite: `66 passed`; full: `351 passed, 2 skipped`; GitHub `Python tests` passed. Fast-forward merge preserved four JapanDino commits and advanced `main` to `b1cd5f5` |
| Research PR policy | Done | `docs/RESEARCH_PR_POLICY.md` defines authorization, identity, Draft/Ready/Merge gates, PR 1–6 timing, pilot/freeze/confirmatory boundaries, PR body, commit evidence, and post-merge rules |
| Commit/PR identity | Enforced by policy/config | Commits use `JapanDino <klim.i.rumyantsev@gmail.com>`; PR author must be GitHub login `JapanDino`; identity is checked independently before commit and before PR |

---

## Status by Component

### Core Gym (`gym/`)

| File | Status | Notes |
|------|--------|-------|
| `notebook.py` | Done | nbformat v4 document editing, stable cell ids, revisions, outputs, Python export |
| `jupyter_kernel.py` | Done | persistent local `ipykernel`; Docker kernel backend with loopback-only ZMQ ports and workspace path translation |
| `notebook_env.py` | Done | real notebook action loop, clean restart-and-run-all, validate, submit, deterministic `type`/`stage`/`thoughts` contract, non-mutating `think`, public/private artifacts |
| `feedback.py` | Done | runtime/contract/checklist/terminal feedback items and generic hidden checklist policy |
| `candidates.py` | Done | candidate records and validation registry |
| `modes.py` | Done | `gym_with_checklist` and `iterative_no_checklist` share the same backend |
| `protocol.py` | Done | canonical action enum, required stage enum, canonical `thoughts`, and `think`; legacy `code` action remains compatible |
| `agent.py` | Done | minimal prompt requires `type`/`stage`; thoughts mode requires `thoughts` and initial `think`/`planning` |
| `llm.py` | Done | OpenAI-compatible, Google/Gemini, and LiteLLM client selection |
| `env.py` | Legacy maintained | old subprocess/Docker environment retained for compatibility tests; rejects `think`/`planning`/`thoughts` because thoughts mode is disabled |
| `executor.py` | Legacy/baseline | Docker/subprocess executor retained for non-notebook baselines |

### Experiments (`experiments/`)

| File | Status | Notes |
|------|--------|-------|
| `run_gym.py` | Done | uses `NotebookGymEnv`, logs notebook/process/private metrics, artifacts to MLflow |
| `run_baseline.py` | Done | single-shot control preserved; prompts require raw-DataFrame pipelines; missing score is not logged as zero |
| `run_multishot.py` | Done | logged as `repeated_single_shot`; prompts require raw-DataFrame pipelines; not the fair checklist control |
| `run_fixed.py` | Done | fixed-transition control preserved; failed submit is not logged as real score 0.0 |
| Paper V1 recorder integration | Done | all four `run_*` entrypoints accept opt-in research artifact flags and link MLflow to experiment/condition/run IDs without changing default execution |
| `run.py` | Done | common single-dataset entrypoint; `--mode all` expands to five separate product runs with shared `batch_id`; `--modes ...` runs a selected batch of up to five modes |
| `compare.py` | Done | handles missing metrics without zero substitution |

### Privacy and Security

| Item | Status | Notes |
|------|--------|-------|
| Hidden test files | Done | not copied into episode workspace; no `test_df` in kernel |
| Hidden score feedback | Done | submit response hides score; score only in private summary/MLflow |
| Local Jupyter sandbox | Limited | real notebook functionality and sanitized env, but not full OS isolation |
| Docker kernel backend | Done | CI builds `Dockerfile.sandbox` and runs Docker integration smoke when Docker is available |
| Agent-visible artifacts | Done | public workspace artifacts exclude hidden score, private checklist coverage, submit failure type, and candidate pickle paths |
| Private evaluator artifacts | Done | private summaries, trajectories, and candidate pickles are stored outside the kernel-visible workspace |

### Tests

| Area | Status |
|------|--------|
| Existing legacy env/executor/agent tests | Passing |
| Jupyter kernel tests | Passing |
| Notebook editing tests | Passing |
| Clean run / validate / submit tests | Passing |
| Checklist privacy/fairness tests | Passing |
| Hidden-test privacy tests | Passing |
| Docker kernel integration | Runs in GitHub Actions after sandbox image build |
| Step-budget semantics | Passing |

---

## Current Verification

Paper V1 protocol cycle (2026-08-12/13):

- `python -m research.validate_protocol research/protocols/protocol_v1.yaml` ->
  valid structure, 120 provisional confirmatory runs, 7 open freeze blockers,
  status `draft_unfrozen`.
- `python -m pytest tests/test_research_protocol.py -q` -> `3 passed`.
- Focused offline suite (episode modes, agent, protocol, experiments) ->
  `66 passed`, one Jupyter path deprecation warning.
- Initial full `python -m pytest -q` -> `267 passed, 2 failed`, one warning; both
  failures were Docker integrations caused by the missing local
  `autovibe-gym-sandbox:latest` image.
- `docker build -f Dockerfile.sandbox -t autovibe-gym-sandbox:latest .` -> passed
  on the second attempt after one external PyPI read timeout.
- The two previously failing Docker integrations were rerun explicitly ->
  `2 passed`, one warning. Together these completed runs cover all 269 collected
  tests; a later monolithic repeat exceeded the local wrapper timeout and is not
  represented as a passing run.
- No API experiments or hidden-test research runs occurred in this protocol
  cycle; the protocol is under review in Draft PR 1.

Paper V1 auditable-manifest cycle (2026-08-13):

- `python -m pytest tests/test_research_run_artifacts.py -q` -> `19 passed`.
- `python -m pytest tests/test_llm.py tests/test_research_run_artifacts.py -q`
  -> `40 passed`, one existing Jupyter path deprecation warning.
- The offline smoke creates an atomic manifest and append-only ledgers, links
  condition/run IDs, redacts secret-shaped values/private notebook fields, and
  classifies unsuccessful outcomes without any API or hidden-test run.
- Provider retry hooks were verified to emit each attempted request and to be
  fail-open for observability, preserving the existing provider return/retry
  behavior.
- Focused experiments/LLM/protocol/research suite -> `70 passed`, one existing
  Jupyter path deprecation warning.
- Full `python -m pytest -q` -> `291 passed`, one existing Jupyter path
  deprecation warning.
- No expensive API experiment or confirmatory run was executed; generated run
  artifacts were confined to pytest temporary directories.
- PR 1 and PR 2 were merged in order on 2026-08-14. PR 2 was rebased onto the
  merged PR 1, passed `289 passed, 2 skipped` locally and the required GitHub
  `Python tests` check, then squash-merged as `67069a3`.

Paper V1 fair-budget cycle (local PR 3 preparation, 2026-08-14):

- Added an opt-in research `EpisodeBudget` with protocol defaults of 64,000
  total reported tokens, 4,096 output tokens per call, 12 logical LLM calls,
  20 code executions, 20 host-tool calls, and 1,800 seconds. Limits are checked
  before the affected provider, execution, or tool call and emit audit events.
- Research-mode repeated single-shot, iterative no-checklist, and gym-with-
  checklist now use the same budget object. Reported reasoning tokens are
  included in total usage; cached-input usage is recorded separately while the
  provider's raw input total remains the budget input count.
- Added one common no-repair submission validator for serializability, raw-row
  prediction, output length, and non-null predictions. Research mode disables
  host-side autofit and host finalization repair/replay; the agent must leave an
  already-fitted candidate that passed clean replay and validation.
- Hidden evaluation is limited to one attempt per terminal candidate outcome.
  A hidden-evaluation failure is terminal and provides no repair feedback to the
  agent. Product-mode retry/autofit compatibility remains unchanged outside the
  opt-in research path.
- Post-outcome LLM summaries are disabled for research runs so they cannot
  consume decision-adjacent budget after the outcome.
- `pytest -q tests/test_research_budget.py tests/test_research_submission.py tests/test_research_run_artifacts.py tests/test_research_protocol.py`
  -> `34 passed`, one existing Jupyter path deprecation warning.
- `pytest -q tests/test_llm.py tests/test_experiments.py tests/test_agent.py tests/test_env.py tests/test_env_protocol.py tests/test_episode_modes.py tests/test_run_summary.py`
  -> `89 passed`, one existing Jupyter path deprecation warning.
- `pytest -q tests/test_notebook_env.py` -> `33 passed, 1 skipped`, one existing
  Jupyter path deprecation warning; the skipped case is the Docker integration
  because Docker was not available to that test process.
- Review follow-up normalizes OpenAI-compatible usage so reasoning tokens remain
  a separately reported subset of completion tokens rather than being charged
  twice. A regression test fixes the expected visible-output/reasoning split.
- A pre-call or pre-execution budget stop in the single-shot control is now a
  recorded `budget_exhausted` terminal outcome and reaches manifest
  finalization without provider access or hidden evaluation.
- Focused review regression -> `77 passed`, one existing Jupyter path
  deprecation warning.
- Full `pytest -q` -> `307 passed, 2 skipped`, one existing Jupyter path
  deprecation warning. `git diff --check` passed.
- Required GitHub `Python tests` passed on PR #65. The additional Docker sandbox
  integration job failed twice on a kernel-readiness timeout, alternating
  between its two integration cases; the production code and focused local
  suites did not reproduce a deterministic failure. CI handshake hardening is
  intentionally not mixed into this research PR without separate approval.
- No API calls, pilot episodes, or hidden-test research runs occurred. PR 3 was
  rebased onto merged `origin/main`, reverified, and published as PR #65.

Paper V1 confirmatory-planner cycle (PR 4, 2026-08-15):

- PR 3 review feedback was answered and its only unresolved thread was resolved
  after commit `385a09f`; required `Python tests` and the Docker sandbox
  integration both passed on the final head.
- PR 3 was fast-forward merged as PR #65 so its existing author/committer
  identities remained `JapanDino <klim.i.rumyantsev@gmail.com>`; merged `main`
  is exactly `385a09f`.
- Added a deterministic confirmatory planner that expands exact dataset/model/
  arm/replicate inputs into unique condition IDs and a canonical immutable plan
  hash. The protocol-sized fixture precomputes 120 conditions in 40 blocks.
- Within every dataset/model/replicate block, A/B/C order uses a portable
  SHA-256-seeded permutation. Input list order cannot change the plan/hash.
- Identical plan creation is an idempotent no-op; a different plan cannot
  overwrite an existing file. The planner rejects unresolved placeholders,
  provisional protocol matrices, pilot budgets, and config/protocol drift.
- Resume reconciliation reads append-only manifests, rejects unplanned or
  duplicate outcomes and broken replacement chains, leaves agent failures as
  terminal outcomes, and queues only declared infrastructure/provider failures
  under the same condition with `rerun_of` and a reason.
- Added JSON schemas for exact plan input and expanded plan output plus CLI/docs
  for offline `build` and `status`. The CLI does not launch episodes.
- Automated review identified and the branch fixed three fail-closed gaps:
  budget status now must equal `frozen`, completed attempts require a declared
  failure category, and concurrent writers publish via an atomic hard link so a
  different preregistered plan cannot be overwritten.
- Focused planner/manifest/protocol/budget/submission suite -> `66 passed`, one
  existing Jupyter path deprecation warning.
- Full `python -m pytest -q` -> `338 passed, 2 skipped`, one existing Jupyter
  path deprecation warning. `git diff --check` passed.
- No exact confirmatory config, real plan, API call, pilot episode, hidden-test
  evaluation, or confirmatory outcome was created or inspected.

Paper V1 preregistered-analysis cycle (PR 5a, 2026-08-15):

- Added a synthetic-only primary pipeline that binds frozen FANU references to
  the immutable plan ID/hash and refuses missing datasets, invalid metric
  directions, or zero denominators.
- Agent failures remain at dummy performance; infrastructure attempts require a
  same-condition terminal replacement and cannot silently enter the primary
  population. Successful outcomes require `valid_submit=true` and exactly one
  hidden evaluation.
- The pipeline emits a machine-readable incomplete-run error before arm effects,
  then pairs H1 `B-A` and H2 `C-B` by dataset/model/replicate when complete.
- Estimates weight dataset-model strata equally. The implementation includes
  paired stratified percentile bootstrap intervals, equal-stratum sign-flip
  permutation tests, exact McNemar valid-submit tests, and Holm correction.
- Raw pairs and per-dataset/model summaries remain in the content-hashed result;
  concurrent writers cannot overwrite a different output, while identical
  reruns are idempotent.
- Automated review identified two P1 provenance gaps: protocol analysis settings
  and FANU reference values were not cryptographically bound to the plan. The
  planner now carries both canonical hashes; analysis recomputes, rejects drift,
  and propagates them into the result. Synthetic tampering regressions cover
  changed seeds/resampling and changed directions/dummy/reference values.
- Follow-up review correctly identified that arm-specific valid-submission
  confidence intervals were required but not frozen. The protocol and pipeline
  now specify and emit 95% Wilson score intervals; paired rate differences still
  use exact McNemar. A separate identity comment cited a nonexistent commit SHA;
  both actual PR commits independently show JapanDino as author and committer.
- Final review added valid fail-closed checks for definitive `git_dirty=false`,
  recomputation of usage/execution totals from disk ledgers, and frozen
  per-arm/dataset/model hidden-score summaries over all agent outcomes and the
  explicitly selection-biased successful-only subset. Two synthetic regressions
  cover dirty provenance and edited manifest totals. A second identity comment
  again cited a SHA absent from both the local object database and GitHub API;
  the actual PR range remains entirely JapanDino-authored/committed.
- Added versioned frozen-reference/result schemas, CLI/docs, and synthetic
  fixtures. Review-fix focused planner/analysis/protocol/artifact suite:
  `66 passed`.
- Full `python -m pytest -q`: `351 passed, 2 skipped`, one existing Jupyter path
  deprecation warning. `git diff --check` passed.
- No API call, pilot, exact confirmatory plan, real manifest, hidden-test score,
  confirmatory result, table, or claim was created or inspected.
- PR #67 passed the final GitHub `Python tests` job in 5m02s. All actionable
  review threads were answered and resolved; repeated identity comments citing
  repository-absent SHAs were closed with local-object, GitHub-API, and exact
  PR-range evidence. The PR was fast-forward merged at `b1cd5f5`, preserving all
  four commits with JapanDino as both author and committer.

Paper V1 protocol-freeze cycle (2026-08-15):

- The authenticated internal OpenAI-compatible endpoint advertised exact IDs
  `deepseek-v4-flash` and `gemma-4-26b`; it did not expose immutable weight
  revisions, so the protocol uses the honest `service-snapshot-2026-08-15`
  label and records request-level provenance. No API secret is committed.
- Result-blind pilots exercised A/B/C for both models on non-confirmatory
  `example_room_occupancy`. The 64k and 128k token limits were rejected because
  iterative runs stopped on pre-call reservation. At 256k, both model-specific
  B checks reached the fixed 12-call ceiling (108283 and 134878 accounted
  tokens) without token-reservation stop. Pilot scores were not inspected for
  selection and pilot outputs cannot enter confirmatory analysis.
- A pilot candidate containing a dynamic `FunctionTransformer` reproduced a
  native Windows access violation during host-side readiness prediction. The
  validator and one-shot hidden gate now execute prediction in an isolated
  worker with timeout/crash containment. Confirmatory Docker mode selects a
  separate ephemeral network-none Docker evaluator rather than a host child.
  Two interrupted infrastructure attempts remain non-overwritten; the
  replacement pilot finalized normally with all ledgers.
- Frozen inputs remove four privileged ground-truth concentrations from Air
  Quality, group diabetes splits by patient and remove both IDs, and remove
  post-contact `duration` from Bank Marketing. Diabetes group overlap is zero.
  Dataset cards record official UCI URLs/DOIs/licenses, local/raw hashes,
  contamination, privacy, and task-framing limitations.
- The fixed HistGradientBoosting reference beats the dummy endpoint on all four
  hidden snapshots. Final immutable hashes are manifest
  `91e4a98523e336929057bb0a4b5b27b54aa2a7b9f2ec0f0057a37989b562d7ac`
  and reference
  `e867776aa650b7445a9fb17d16686e176dca8b14cc0c1460426708fe162e0570`.
- `python -m research.validate_protocol research/protocols/protocol_v1.yaml` ->
  valid 120-run structure, zero open freeze blockers, status `frozen`.
- Focused freeze/protocol/submission/notebook suite -> `48 passed`, one existing
  Jupyter path warning, in 343.49 seconds.
- Full `python -m pytest -q` -> `361 passed`, one existing Jupyter path
  deprecation warning, in 425.44 seconds. `git diff --check` passed.
- No real confirmatory plan was generated; no confirmatory episode, arm effect,
  p-value, confidence interval, or scientific conclusion was inspected.

Last local run:

```bash
python -m pytest
```

Result after deterministic action observability: `236 passed` (one sklearn
`InconsistentVersionWarning` from the Docker notebook pickle smoke).
Current deterministic action observability cycle:

- Canonical action schema now uses `type`, required `stage`, canonical
  `thoughts`, and non-mutating `think`.
- `python -m pytest` -> `236 passed`, one existing sklearn warning.
- Dashboard TypeScript build + Vite production build passed via bundled Node
  runtime (`tsc -b`, `vite build`).
- Browser smoke on `http://127.0.0.1:8010/runs/live_codex_stage`: Run Detail
  header showed `Этап = Анализ валидации`; Trajectory rendered `think` and
  type/stage badges; Thoughts tab rendered scratchpad entries; desktop and
  mobile viewport console checks had no warnings/errors.
- Follow-up launcher check after a real dashboard run: selected model records
  now explicitly set `LLM_PROVIDER` (`openai`, `google`, or `litellm`) so a
  `.env` Gemini default cannot route OpenAI-compatible dashboard models through
  the Google SDK; `python -m pytest tests/test_dashboard.py -q` -> `17 passed`.
Current model-registry config cycle:

- `.env.example` and local `.env` no longer contain LLM model settings or LLM
  API keys; model names, provider, endpoint URL, and per-model API key live in
  the shared model registry.
- Added `python -m experiments.models` for CLI model registry management.
- `experiments.run`, `run_matrix`, and the `run_*` scripts require `--model` and
  resolve it as a model id/name from the registry.
- Gemini models no longer require or send Base URL; LiteLLM receives the
  per-model key through the registry-backed runtime path.
- `python -m pytest tests/test_llm.py tests/test_dashboard.py tests/test_experiments.py -q` -> `64 passed`.
- Dashboard TypeScript build (`tsc -b`) and Vite production build passed.
Current dashboard multi-select cycle:

- `python -m pytest tests/test_dashboard.py tests/test_experiments.py` -> `39 passed`
- dashboard TypeScript build + Vite production build via bundled Node runtime
- Browser smoke on `/new`: `All modes` is absent and run types are selected as
  separate product modes.
Current dashboard responsive-hardening cycle:

- Backend API healthy on `http://127.0.0.1:8000/api/health`; Vite dev server
  running on `http://127.0.0.1:5174/new`.
- dashboard TypeScript build + Vite production build via bundled Node runtime.
- CSS hardened for compact top navigation, responsive New Run stacking, card/grid
  min-width behavior, wrapped filter/setting rows, dataset actions, compare picks,
  preview values, and chart bars so narrow and desktop breakpoints avoid clipped
  controls and horizontal layout drift.
Current dashboard five-mode selection cycle:

- New Run includes Iterative again, allows selecting up to 5 run types, removes
  the recommendation badge, and labels the interactive modes as Flexible gym and
  Fixed gym.
- Shared product-mode metadata, common `experiments.run --mode all`, dashboard
  batch launch validation, frontend types, and tests now use the same five-mode
  product set.
Current dashboard environment-badge cycle:

- New Run run-type cards now show only two badges: `Среда` for Iterative,
  Flexible gym, and Fixed gym; `Без среды` for Single-shot and
  Repeated single-shot.
- dashboard TypeScript build + Vite production build via bundled Node runtime.
Current dashboard budget-tooltip cycle:

- Removed the inline `local — длиннее, cloud — экономнее` hint from the New Run
  budget preset field.
- Added Problems-style `?` tooltips to New Run budget parameter labels.
- dashboard TypeScript build + Vite production build via bundled Node runtime.
Current common-runner failure-propagation cycle:

- `experiments.run` now exits with the first failed child return code after
  printing the batch summary, even when `--stop-on-failure` is not set.
- `python -m pytest tests/test_experiments.py` -> `27 passed`.
Current dashboard run-detail tabs cycle:

- Dashboard tab menus no longer have a 1px vertical overflow: the divider line is drawn inside the tab strip and tab buttons no longer use a negative bottom margin.
Current dashboard/product-mode label cycle:

- Dashboard mode labels now display `Flexible gym` wherever the gym product mode
  was previously shortened to `Flexible`.
- Fixed-transition product mode display text is shortened to `Fixed gym` in the
  dashboard and shared experiment matrix labels.
- `dashboard/web`: TypeScript build + Vite production build passed.
- `.venv/bin/python -m pytest tests/test_experiments.py` -> `27 passed`.
Current dashboard sidebar cleanup cycle:

- Removed the bottom sidebar team/local-mode block from the dashboard shell.
- Moved the sidebar collapse toggle down to the bottom edge of the simplified
  sidebar.
Current dashboard logo cycle:

- Replaced the old H-like sidebar logo mark with a diagonal dumbbell mark
  matching the yellow-square fitness reference while avoiding letter shapes.
- `dashboard/web`: TypeScript build + Vite production build passed.
- Browser smoke captured `.sidebar-logo`; only existing dev warnings/favicon 404
  appeared in console.
Additional checks:

- `python -m experiments.run --dataset-dir datasets/demo/prepared --mode all --model fake-model --dry-run`
- `python -m experiments.run_all_modes_matrix --datasets datasets/demo/prepared --models fake-model --dry-run`

The Docker-backed notebook integration test ran locally in this Windows
workspace and passed. GitHub Actions remains the source of truth for the Linux
sandbox image build.

---

## H200 Recon Findings (2026-06-01)

First full run of all modes × both cloud models (`deepseek-v4-flash`, `gemma-4-26b`)
on the H200 server. Cloud API (`llm.letovo.site`) exposes only these two models —
no Qwen yet (curator's Qwen3-32B suggestion needs local vLLM serving).

Fixed in this PR:

- **OpenBLAS / OpenMP thread exhaustion.** The subprocess sandbox env and the
  local Jupyter kernel env set no thread caps. On the many-core server this aborts
  model training with `OpenBLAS: Memory allocation failed` (subprocess backend) and
  xgboost ctypes `DataIter` crashes (notebook kernel). Now capped via
  `thread_limit_env()` in `executor.py`, `_minimal_kernel_env()`, and the Docker
  kernel `--env` block (overridable with `AUTOVIBE_SANDBOX_THREADS`).
- **`run_fixed.py` CLI parity.** It rejected `--max-steps` (used by every other
  runner and the matrix), crashing the fixed-transition runs with exit 2. Added.
- **scikit-learn replay skew.** Candidate pickles were written in the sandbox image
  (sklearn 1.7.2) and read on the host venv (1.8.0) → `InconsistentVersionWarning`.
  Pinned `scikit-learn==1.7.2` so the image and host resolve identically.

Gym `test_metric=null` — root-caused and fixed (next PR, branch
`dev/claude/gym-good-score`):

- **Cause 1 — action parsing.** `gemma-4-26b` wraps its JSON in chat-template
  tool-call tokens (`{...}<tool_call|>`, `<|tool_call>call:{...}`). The strict
  parser required the text to end in `}`, fell through to the legacy code
  fallback, and dumped the raw action text into a notebook cell → SyntaxError
  loop, so the agent never reached validate/submit. Fixed with robust extraction
  (strip wrapper tokens, balanced-brace JSON scan, `strict=False`).
- **Cause 2 — brittle clean run.** restart_and_run_all re-runs the whole messy
  notebook; missing libs (`seaborn`), slow GridSearchCV (per-cell timeout), and
  cross-cell `NameError`s from deleted cells made every clean run fail. Mitigated
  by installing matplotlib/seaborn, raising the per-cell timeout to 120s, and
  steering the prompt to finalize early with a small search.
- **Cause 3 — strict finalize + label skew.** Forced submit needed a prior
  validate; agents that built a good model but mismanaged the protocol got null.
  Added host-side `NotebookGymEnv.finalize()` (live-kernel fallback before any
  kernel-wiping restart) and label dtype coercion in validate/submit.
- **Verified:** a dirty-kernel, label-encoded model now finalizes to
  `final_test_metric=0.7247` f1_macro on `student_dropout` (vs baseline 0.739).

Single-shot / repeated multishot failures — addressed in this PR (branch
`dev/claude/oneshot-multishot-fix`):

- **Raw hidden-test input mismatch.** Baseline/multishot prompts now require a
  fitted scikit-learn `Pipeline` / `ColumnTransformer` assigned to `model`, so
  `model.predict(raw_df)` works on raw validation and hidden-test rows instead
  of relying on notebook-side preprocessing state.
- **Label dtype mismatch.** The label-coercion scoring helper is now shared by
  `NotebookGymEnv`, `run_baseline.py`, and `run_multishot.py`, so label-encoded
  integer predictions can be scored against string/categorical targets.
- **Joblib worker crashes in sandbox.** The subprocess/Docker executor forces
  sequential joblib/loky execution (`JOBLIB_MULTIPROCESSING=0`,
  `LOKY_MAX_CPU_COUNT=1`), preventing `n_jobs=-1` searches from failing when
  process spawning is restricted.
- **Pending verification:** rerun the H200 matrix to confirm DeepSeek
  baseline/multishot no longer end as `submit_failed`.

Still open (need server-side iteration):

- **Sandbox image name.** Default is `autovibe-gym-sandbox:latest`; the server only
  had `autovibe-gym:latest` built, and rootless Docker can't pull from docker.io.
  Server setup task: build the image under the expected tag.

## Web Dashboard (`dashboard/`, branch `dev/claude/web-dashboard`)

Local control panel, separate from `gym/`. Reuses the project `.venv`.

- **Backend** (`dashboard/server`, FastAPI): reads runs from MLflow and parses
  episode artifacts into notebook/trajectory/checklist/errors/logs (per-item
  checklist closure reconstructed by replaying public notebook events through the
  gym's own `NotebookChecklist`); datasets CRUD + CSV upload over `datasets/`;
  models JSON registry seeded from `.env` with OpenAI-compatible health probe;
  run launcher spawning `run_baseline/run_multishot/run_gym` subprocesses (shared
  MLflow store), live status + process-log tail + stop.
- **Frontend** (`dashboard/web`, Vite+React+TS): all 8 screens built to the
  T-Bank design tokens — Dashboard, New Run, Runs, Run Detail (5 tabs), Compare,
  Datasets (+detail/upload/delete), Models, Settings. Light/dark theme + accent.
- **Execution modes:** each run picks where the gym executes — **local** (on the
  machine running the backend; only the LLM call is remote — works off-VPN) or
  **server (SSH)** (gym + kernels run on the GPU server, results synced back via
  rsync; configured in Settings). The site can also be served entirely from the
  server (single-app mode) when a port is reachable.
- **Models:** registry seeded with the team's gemma/deepseek on the shared LLM
  server; any OpenAI-compatible endpoint (e.g. Cerebras/Groq/Gemini) can be added.
  Header pill shows LLM server reachability.
- **Live updates:** runs launch into a known `data/runs/<id>/workspace` dir; the
  gym already flushes public artifacts after every step, so while a run is in
  progress the dashboard reads that dir directly — step counter, checklist
  coverage, notebook cells, trajectory and logs all advance live (detail polls
  every 2.5s, running runs enriched via `episode_progress`). MLflow is used for
  finished runs.
- **Verified:** `npm run build` clean; FastAPI serves over HTTP; Vite dev proxies
  `/api`; real MLflow runs/datasets render; launcher builds correct commands;
  simulated mid-run workspace confirms live step/checklist/notebook/logs reads.
- **Hardened after merge:** Windows/default Python detection now resolves the
  repo `.venv\Scripts\python.exe` (with `sys.executable` fallback), single-shot
  and repeated single-shot launches use the correct planned step counts, MLflow
  mode/progress/status mapping handles `baseline_single_shot` and repeated
  attempts, placeholder zero scores from failed submits are hidden, checklist
  detail/list coverage both fall back to authoritative MLflow coverage when
  episode artifacts are absent, and the run detail donut uses that authoritative
  coverage value. The responsive shell now switches to compact top navigation on
  mobile so the dashboard has no page-level horizontal overflow.
- **Dataset Center expansion:** `/datasets` now manages raw and prepared data
  with search/filter/sort, staged file and URL uploads, safe archive extraction,
  table preview, raw-table splitting, prepared-file mapping, rich
  `dataset_config.json`, compatible `prepared/meta.json`, editable sources and
  agent notes, and backward-compatible display for old prepared datasets.
- **Dataset Center polish:** dataset cards render as a one-column list, the UI is
  localized to Russian while preserving common ML terms (`target`, `seed`,
  `raw`, `train/val/test`, `Target column`), dataset suite/group metadata is
  removed from dashboard/project flows, empty sources display `-`, and example
  configs now carry the repository creation timestamp plus UCI source metadata.
- **Dataset Center PR #34 hardening:** URL downloads now reject localhost/private
  targets and unsafe redirects, archive extraction budgets count decompressed
  gzip bytes, failed create-from-config attempts clean their temporary dataset
  root, root-format legacy datasets keep root `meta.json` edits, JSONL raw
  uploads are covered by tests, `/datasets` remains compatible with the new
  `/problems` navigation, and CI dataset preparation uses the current
  `prepare_datasets.py` CLI.
- **Verified after hardening:** backend API smoke confirms `/api/health`,
  `/api/runs`, and `/api/runs/{id}/checklist` agree on `11/12` and `0.88` for a
  legacy MLflow run without episode events; browser smoke on desktop/mobile
  confirms `11/12`, `88%`, no stale `92%`, no console errors, and no page-level
  horizontal overflow.
- **Visual fix:** active trajectory rows now render the "agent is executing"
  spinner inside the same marker column as step icons; the connector from the
  previous step ends at the spinner's top center.
- **Run:** `dashboard/server/run.sh` (API :8000) + `cd dashboard/web && npm i && npm run dev` (:5173).

## Blocked / Needs Decision

- The freeze decisions and pilot are complete, but confirmatory execution is
  blocked until this branch passes the full regression suite, is reviewed and
  merged, and merged `main` receives immutable tag
  `paper-v1-experiment-freeze`.
- The second annotator and adjudicator are nominees only. Checklist annotation
  cannot start until each gives written consent; this does not block the model
  run matrix.
- No frozen data, trajectory bundle, GitHub Release, Zenodo record, arXiv post,
  or journal submission is externally published. Those remain separate actions
  after the study/manuscript and explicit owner approval.
- Local Docker CLI is available in this Windows workspace as of 2026-06-02; the
  Docker-backed notebook integration test passed locally. GitHub Actions still
  verifies the Linux sandbox image path.
- Existing `GymEnv` remains for compatibility, but new iterative experiments
  should use `NotebookGymEnv`.
- Repository owner still needs to confirm the `main` ruleset requires the
  `Python tests` status check before merge if connector/API access cannot verify
  rulesets.

---

## Next Actions

1. [x] Completed full local regression of the frozen protocol, exact plan,
       dataset cards, pilot evidence, and crash-containment changes; human PR
       review remains a merge gate.
2. [x] PR 1 and PR 2 were reviewed through their required gates and merged in
       order; PR 2 passed local and required GitHub verification after rebase.
3. [x] Rebased, reviewed, reverified, and fast-forward merged PR #65: one fair
       global budget across A/B/C, research-mode no-autofit validation, and one
       hidden evaluation per terminal agent outcome. Merged `main` is `385a09f`.
4. [x] Reviewed and fast-forward merged PR #66: complete
       condition matrix/hash, blocked randomization, resume/idempotency,
       duplicate rejection, replacement queue, schemas, CLI, and acceptance checks.
5. [x] Published, reviewed, and fast-forward merged PR #67: preregistered FANU,
       completeness, paired analysis, valid-rate intervals, score sensitivities,
       ledger reconciliation, and synthetic fixtures before any outcome inspection.
6. [x] Completed a labeled result-blind availability/budget pilot after PR 5a;
       selected 256k/12 calls and kept all pilot outputs outside confirmatory data.
7. [ ] Merge the freeze PR, create and verify tag
       `paper-v1-experiment-freeze`, then execute only the already-generated
       immutable 120-condition plan. The plan/config/protocol hashes reconcile
       and every condition binds execution commit `b4e3da2`.
8. [ ] Keep the earlier H200 notebook-era matrix rerun and experiment-report
       refresh as product/pilot validation, clearly separated from Paper V1.
9. [x] Existing product PRs were merged to `main`; TZ/PROTOCOL/EXPERIMENT_REPORT
       were synchronized before this Paper V1 cycle.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-08-15 | Closed the three actionable PR #69 review gaps: Arm A now explicitly couples Docker execution to Docker candidate prediction; the evaluator returns a strict typed JSON scalar vector instead of host-unpickled result objects; and the exact result-blind confirmatory plan now exists before the freeze tag. Also bound API temperature 0.4 to actual requests and decoding hashes, normalized prompt-template hashing across datasets, added explicit A/B/C manifest identity, and reduced the default provider retry limit to the preregistered three. Execution commit `b4e3da2` passed `372 passed`; plan `plan_1c2ad2a922165e0d27471ae7` contains 120 pending conditions with hash `1c2ad2...b593`. No confirmatory outcome or external publication was created. |
| 2026-08-15 | Prepared the Paper V1 protocol freeze after nine completed result-blind pilot episodes. Froze authenticated internal `deepseek-v4-flash`/`gemma-4-26b`, 256k tokens with 12 calls and 0-ruble/no-paid-fallback policy, four leakage-reviewed dataset snapshots, FANU dummy/reference values, storage choice, annotator nominees, and TMLR-style manuscript format. Added dataset cards, endpoint/pilot snapshots, immutable manifest/reference hashes, and a no-overwrite freeze generator. Pilot-discovered native candidate prediction crashes are now contained; confirmatory prediction runs in an ephemeral network-none Docker evaluator with no host secrets. Focused suite `48 passed`; full suite `361 passed`; Docker prediction smoke passed. No confirmatory plan/outcome or external publication was created. |
| 2026-08-15 | Fast-forward merged Paper V1 PR 5a (#67) at `b1cd5f5` after final GitHub `Python tests` success and resolution of all actionable review threads. Preserved four commits with author/committer `JapanDino`. The frozen analysis now fails closed on incomplete/dirty/drifted inputs, binds protocol/reference hashes, reconciles disk ledgers, and preregisters FANU H1/H2, Wilson valid-rate intervals, McNemar/Holm, and all-outcome/successful-only summaries. No API, pilot, real manifest, confirmatory outcome, or scientific claim was created or inspected. |
| 2026-08-15 | Fast-forward merged Paper V1 PR 4 (#66) at `1b12017` after all review threads and GitHub checks passed, preserving author/committer `JapanDino`. Published PR 5a as #67: fail-closed completeness, frozen FANU references, paired H1/H2, equal-stratum bootstrap/permutation, exact McNemar, 95% Wilson rate intervals, all-outcome/successful-only score summaries, ledger reconciliation, Holm, raw/stratum outputs, schemas, and immutable content-hashed publication. Review fixes cryptographically bind protocol and reference payloads to the plan. Synthetic-only focused suite after fixes: `66 passed`; full suite: `351 passed, 2 skipped`. No API/pilot/confirmatory outcome was created or inspected. |
| 2026-08-15 | Fast-forward merged Paper V1 PR 3 (#65) at `385a09f`, preserving author/committer `JapanDino`; all final GitHub checks passed. Published PR 4 as #66: deterministic immutable condition planning, 120-condition acceptance fixture, SHA-256 blocked A/B/C ordering, concurrent-safe no-overwrite plan creation, exact frozen-budget/category gates, manifest resume/duplicate checks, and infrastructure-only replacement queue. Review-fix focused suite: `66 passed`; full suite: `338 passed, 2 skipped`. No plan with real identities and no API/pilot/confirmatory run was created. |
| 2026-08-14 | Merged Paper V1 PR 1 (`6bb75d1`) and PR 2 (`67069a3`) in order after required checks. Published PR 3 as #65: pre-call global episode budgets and audit events, common no-autofit submission validation, one-shot hidden evaluation without repair feedback, reasoning/cached-token observability, and research-only suppression of post-outcome LLM summaries. Review follow-up removed reasoning-token double counting and ensured budget exhaustion still finalizes the single-shot manifest. Full offline suite: `307 passed, 2 skipped`; required GitHub `Python tests` passed; the separate Docker integration job remains flaky at kernel readiness. No API/pilot/confirmatory run occurred. |
| 2026-08-12 | Codified mandatory authorship and publication rules: every commit must use `JapanDino <klim.i.rumyantsev@gmail.com>`, every PR must be authored by GitHub login `JapanDino`, and stage/commit/push/PR/merge require separate explicit authorization. Added `docs/RESEARCH_PR_POLICY.md` with Draft/Ready/Merge timing, PR 1–6 sequencing, preregistered PR 5a before freeze, pilot/confirmatory gates, PR-body evidence requirements, and post-merge provenance rules. |
| 2026-08-12 | Prepared the unfrozen Paper V1 research package from freshly fetched `origin/main` commit `1504cc0`: Phase 0 evidence-backed code audit, canonical machine-readable protocol, hypotheses, analysis plan, failure/retry policy, dataset/model selection gates, offline validator, and focused tests. Confirmed current per-call token semantics, host autofit in single/repeated controls, separate step/tool accounting, private hidden score, three-attempt hidden failure loop, and 8-vs-12 checklist documentation drift. No runner behavior, API experiment, commit, push, or PR was performed. |
| 2026-06-08 | Gym now beats single-shot (experiment-validated, gemma-4-26b): (1) validation-improvement nudge — after `validate`, NotebookGymEnv reports best-so-far + remaining budget and pushes the agent to beat its own baseline (honest, val-split only, feedback modes only); flips student_dropout from −0.004 (gym lost) to +0.013 over single-shot. (1b) unknown-cell-id guard — targeting a non-existent cell is now a recoverable blocker instead of a KeyError that crashed the whole episode, restoring valid-submit rate to 1.0. Measured-but-rejected: per-class-recall diagnostic and candidate-list nudge both hurt (extra feedback verbosity inflates the trajectory → more clean-run failures) |
| 2026-06-06 | Tightened run-summary UX: the prompt remains English-only, normalization no longer chops already-short section bodies mid-sentence, and the Run Detail «Саммари решения» card now renders parsed summary sections (plus inline code) as a structured two-column report instead of raw markdown paragraphs |
| 2026-06-06 | Post-run self-summary: once the model solved the task (reached a final submit for gym/fixed — even if the hidden test rejected it — or produced a usable candidate for single/repeated) one extra best-effort LLM call asks the model to summarize its own solution; saved as `run_summary.json` (`gym/run_summary.py`), served via `GET /runs/{id}/summary` + `hasSummary` on `get_run`, and rendered as a standalone report card (accent side-stripe, neutral surface — deliberately not styled like a step thought) above a «Ход рассуждений по шагам» section on the «Мысли» tab, which now shows for any run with a summary even when thoughts mode is off (old/unsolved runs without either stay hidden). Privacy preserved: the summarizer only sees the conversation it already had, never the hidden test score |
| 2026-06-06 | Dashboard launcher now passes explicit `LLM_PROVIDER` from the selected model provider, preventing OpenAI-compatible models from inheriting a Gemini `.env` default |
| 2026-06-06 | LLM model configuration moved out of `.env` into the shared model registry with CLI management, required `--model` registry resolution, provider-specific Gemini/LiteLLM runtime env, and no LLM keys in `.env` |
| 2026-06-06 | Hardened post-submit run-summary generation: stricter four-section retrospective prompt in English, lower token cap, and normalization that strips reasoning/draft/checklist scaffolding both when generating new summaries and when reading already-saved `run_summary.json` artifacts |
| 2026-06-06 | Post-submit self-summary is now explicitly retrospective and grounded in the actual submitted solution code (`final_notebook.py` / generated baseline code), while the dashboard thoughts toggle/launcher scope is limited to Iterative and Gym rather than Fixed |
| 2026-06-06 | Deterministic action observability: agent JSON actions now use canonical `type`, required ordered `stage`, canonical `thoughts` in thoughts mode, and non-mutating `think`; NotebookGymEnv/GymEnv enforce the contract, artifacts/API/dashboard expose current stage and thoughts, docs/tests updated |
| 2026-06-06 | Added LLM-agent ML toolbox to requirements.txt + pyproject.toml: catboost, seaborn, plotly, optuna, shap, imbalanced-learn, category_encoders, statsmodels, tabulate — all missing from the venv, all commonly reached for by agents; seaborn was already listed in requirements.txt but never installed (PR #53) |
| 2026-06-06 | Dashboard runs table: added `.truncate` CSS (max-width 220px, ellipsis) on model and dataset columns to eliminate horizontal micro-scroll caused by long model names after chevron removal (PR #50) |
| 2026-06-06 | Dashboard runs table: removed the chevron `›` column at row end — rows are visibly clickable without it (PR #49) |
| 2026-06-05 | Found "sweet-spot" weak model for gym-vs-single-shot gap experiments: `llama-3.1-8b-instant` on Groq (val≈0.661 single-shot); added to models.json (gitignored). OpenRouter free limit (50 req/day) inadequate for gym runs. Groq TPM cap (~6000) handled by PR #47 max-tokens clamp |
| 2026-06-05 | Failed runs now expose all available info across every mode: repeated single-shot writes a full multi-attempt episode (every attempt's code + error visible in Notebook/Trajectory/Errors/Logs, not just the best), the run record carries `failReason`/`finalStatus` from the runner's `null_reason`/`final_status`, and the detail page always shows the fail banner with the status label |
| 2026-06-05 | Dashboard launcher clamps `--max-tokens` to the selected model's `maxTokens` cap, so providers with tight per-minute token limits (e.g. Groq free ~6000 TPM) don't 413 when the New Run form leaves the default high |
| 2026-06-05 | Dashboard tabs visual fix: removed the 1px vertical overflow in tab menus by moving the divider into the tab strip and dropping the negative tab margin |
| 2026-06-04 | Dashboard logo refresh: replaced the H-like yellow-square mark with a diagonal dumbbell AutoVibe mark |
| 2026-06-04 | Dashboard sidebar cleanup: removed the bottom `Команда / локальный режим` block and its unused styles |
| 2026-06-04 | Normalized product-mode display labels: dashboard short labels now show `Flexible gym`, and fixed transitions display as `Fixed gym` in dashboard and shared matrix metadata |
| 2026-06-02 | Dashboard trajectory visual fix: aligned the live "агент выполняет шаг…" spinner with the timeline marker column and adjusted the connector so the grey line meets the spinner's top center |
| 2026-06-02 | Agent thoughts/scratchpad: initial visible-thoughts support for gym/iterative runs with `--enable-thoughts`, persistent `scratchpad.json`, context reinjection, dashboard New Run toggle, and a «Мысли» tab. |
| 2026-06-03 | Propagated child runner failures from `experiments.run`: batch summaries still print, but the wrapper exits non-zero when any selected product mode fails |
| 2026-06-03 | Cleaned up New Run budget controls: removed the budget-preset subhint and added Problems-style tooltip hints to budget parameter fields |
| 2026-06-03 | Added two-state environment badges to New Run mode cards: `Среда` for the three environment-backed modes and `Без среды` for the two non-environment modes |
| 2026-06-03 | Restored Iterative as a selectable product run type, raised dashboard/common-run batch selection to 5 modes, renamed Gym to Flexible gym and Fixed transitions to Fixed transitions gym, and removed the New Run recommendation badge |
| 2026-06-03 | Hardened dashboard responsive layout across core routes: mobile hides the desktop sidebar toggle, New Run stacks earlier, grids/cards/settings/filters/dataset actions can shrink or wrap safely, compare labels and preview values wrap, and chart bars no longer force overflow |
| 2026-06-03 | Replaced the dashboard `All modes` launch card with multi-select run types; multi-run launches use `batch` + `--modes ...` while preserving the CLI `--mode all` path |
| 2026-06-03 | Added first-class `all` run orchestration: shared product-mode metadata, `experiments.run --mode all`, matrix batch metadata, compare columns/sort for `requested_mode`/`batch_id`/`mode_label`, dashboard `All modes` and `Fixed transitions` launch options, and responsive New Run grid fix |
| 2026-06-03 | PR #34 Dataset Center hardening: fixed CI dataset preparation, restored finite upload limits, blocked localhost/private URL downloads and unsafe redirects, enforced gzip decompressed-size limits, made dataset creation atomic, preserved legacy root `meta.json` edits, added JSONL/SSRF/cleanup regressions, and kept `/datasets` route compatibility after the `/problems` UI rename |
| 2026-06-02 | Dataset Center polish: one-column dataset cards, Russian UI with common ML terms preserved, dataset suite/group metadata removed from project flows, example configs now include repository-created timestamp and UCI sources, and empty sources display `-` |
| 2026-06-02 | Dataset Center full workflow: backend staged uploads/URL downloads/safe archive extraction/table preview/create-from-config/config editing, React Dataset Center search/filter/sort, full creation wizard, seven-tab detail page, docs and backend tests |
| 2026-06-02 | Single-shot/repeated now show code + checklist coverage in the dashboard: the legacy runners emit a synthesized episode (solution.ipynb, notebook_events, feedback_trace, summary) into `--workspace-dir` and log `checklist_coverage` measured from the generated code, so Notebook/Trajectory/Checklist tabs populate for these modes too |
| 2026-06-02 | Single-shot/repeated now produce a score locally: raised legacy executor timeout 60→300s, prompts require a fitted predict-ready model, and the runner auto-fits an unfitted submitted model before scoring (gpt-oss-120b left it unfitted). Verified single_shot=0.931 f1 on example_dry_bean |
| 2026-06-02 | Dashboard local-execution fix: single-shot/repeated use the legacy executor which defaulted to `docker` from .env → "no candidate" on a Mac without Docker. Local launches now force the in-process `subprocess` executor + local kernel (env `AUTOVIBE_DASHBOARD_EXECUTOR`) |
| 2026-06-02 | Dashboard visual polish: fixed sidebar (position:fixed), dumbbell logo replacing the «A» mark, cleaner gear/trash icons, and a rebuilt trajectory timeline — per-step icon badges by step type (add/edit/delete/restart/run/validate/submit) with opaque fills and a clean connector line |
| 2026-06-02 | Dashboard checklist consistency: tab count uses the recorded `checklist_coverage` (single source of truth, matches the run banner) and exactly that many items render green; aligned the live-banner count to the same formula |
| 2026-06-02 | Dashboard execution modes: per-run selector «на сервере (SSH) / на компьютере» on New Run (overrides the global default); local mode runs gym on the machine and calls the remote LLM (works off-VPN) |
| 2026-06-02 | Dashboard remote execution: run the gym on the GPU server over SSH while the site stays local (`services/remote_exec.py`: ssh/rsync launch + artifact sync + run-summary parse; key auth or optional expect password); configurable in Settings with a connectivity probe |
| 2026-06-02 | Dashboard single-app server mode: FastAPI serves the built SPA (one process) so the whole dashboard can run on the server; added `serve.sh` and deploy docs |
| 2026-06-02 | Dashboard live updates: launches write to a known workspace dir and the backend reads in-flight artifacts, so step/checklist/notebook/trajectory/logs advance during a run (2.5s polling); models registry seeded with team gemma/deepseek; header pill switched to LLM "Сервер онлайн/офлайн" |
| 2026-06-02 | Dashboard polish: fixed Windows Python discovery for run launches, stabilized single-shot/repeated launch progress, reconciled checklist list/detail coverage for legacy MLflow runs, hid placeholder failed-submit scores, pinned `scikit-learn==1.7.2` in `pyproject.toml`, added dashboard regression tests, and fixed mobile layout overflow |
| 2026-06-01 | Dashboard run fixes: dedup the MLflow twin of a live launch by run-name; reconcile orphaned 'running' metas (server reload mid-run) against MLflow; live per-second duration on client; cap launch threads (OMP/BLAS/MKL + AUTOVIBE_SANDBOX_THREADS, sequential joblib) to stop CPU/fan spikes; `run.sh` reload off by default (gym `.py` artifact writes were restarting uvicorn mid-run) |
| 2026-06-01 | Added local web dashboard (`dashboard/`): FastAPI backend (MLflow runs, episode-artifact parsing, datasets/models CRUD, subprocess run launcher) + Vite/React/TS frontend with all 8 screens on the T-Bank design system |
| 2026-06-01 | Fixed single-shot/repeated multishot H200 failure modes: raw-input pipeline prompts, shared label-coercion scoring, sequential joblib in the subprocess/Docker executor |
| 2026-06-01 | Fixed gym `test_metric=null`: robust action parsing (tool-call tokens), host-side `finalize()` live-kernel fallback, label-encoding coercion in validate/submit, viz libs + per-cell timeout + prompt steering; verified 0.7247 on student_dropout |
| 2026-06-01 | H200 recon: capped BLAS/OMP threads in sandbox+kernel, added `run_fixed --max-steps`, pinned scikit-learn==1.7.2; documented open gym-submit issue |
| 2026-05-29 | Hardened notebook privacy artifacts, Docker kernel path/port handling, step-budget blocking, and deterministic CI sandbox image build |
| 2026-05-29 | Implemented ContainerJupyterKernelBackend: Docker sandbox with internal network, read-only rootfs, and dropped capabilities |
| 2026-05-29 | Rebasing Jupyter branch on updated `origin/main` and preserving LiteLLM/Groq provider support |
| 2026-05-29 | Added real Jupyter `.ipynb` + persistent kernel backend with notebook actions |
| 2026-05-29 | Added clean `restart_and_run_all`, host-controlled `validate`, candidate registry, and submit gate |
| 2026-05-29 | Split feedback channels and replaced dataset-specific checklist hints with generic selective hints |
| 2026-05-29 | Hid hidden test metric from agent-facing messages, feedback traces, and notebook outputs |
| 2026-05-29 | Added `iterative_no_checklist` as fair Jupyter control and renamed multishot logging to `repeated_single_shot` |
| 2026-05-29 | Added Jupyter dependencies and tests for kernel, notebook edits, clean replay, validation, privacy, and fairness |

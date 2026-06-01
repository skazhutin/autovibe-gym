# AutoVibe Gym - Live Status

**Last updated:** 2026-06-01 (gym test_metric=null root-caused and fixed: robust action parsing + host-side finalize + label coercion)
**Phase:** Hardening after first full H200 recon. All 4 run types × 2 cloud models executed on `example_student_dropout` (+ `example_room_occupancy`); fixing the blockers found.

---

## Current Sprint Goal

Harden the merged real Jupyter + Docker-backed kernel environment with
behavioral privacy tests, deterministic PR CI, and minimal fixes for discovered
test, sandbox, and logging gaps.

---

## Status by Component

### Core Gym (`gym/`)

| File | Status | Notes |
|------|--------|-------|
| `notebook.py` | Done | nbformat v4 document editing, stable cell ids, revisions, outputs, Python export |
| `jupyter_kernel.py` | Done | persistent local `ipykernel`; Docker kernel backend with loopback-only ZMQ ports and workspace path translation |
| `notebook_env.py` | Done | real notebook action loop, clean restart-and-run-all, validate, submit, public/private artifacts |
| `feedback.py` | Done | runtime/contract/checklist/terminal feedback items and generic hidden checklist policy |
| `candidates.py` | Done | candidate records and validation registry |
| `modes.py` | Done | `gym_with_checklist` and `iterative_no_checklist` share the same backend |
| `protocol.py` | Done | notebook actions added; legacy `code` action remains compatible |
| `agent.py` | Done | prompt updated for notebook actions and clean-run/validate/submit contract |
| `llm.py` | Done | OpenAI-compatible, Google/Gemini, and LiteLLM client selection |
| `env.py` | Legacy maintained | old subprocess/Docker environment retained for compatibility tests |
| `executor.py` | Legacy/baseline | Docker/subprocess executor retained for non-notebook baselines |

### Experiments (`experiments/`)

| File | Status | Notes |
|------|--------|-------|
| `run_gym.py` | Done | uses `NotebookGymEnv`, logs notebook/process/private metrics, artifacts to MLflow |
| `run_baseline.py` | Done | single-shot control preserved; missing score is not logged as zero |
| `run_multishot.py` | Done | logged as `repeated_single_shot`; not the fair checklist control |
| `run_fixed.py` | Done | fixed-transition control preserved; failed submit is not logged as real score 0.0 |
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

Last local run:

```bash
python -m pytest --cov=gym --cov=experiments --cov=scripts --cov-report=term-missing --cov-report=xml --cov-fail-under=70
```

Result after hardening changes: `154 passed, 2 skipped`, coverage `75.26%`.
The two skipped tests are Docker integration tests skipped only in the local
Windows environment because Docker CLI is unavailable; GitHub Actions now builds
`autovibe-gym-sandbox:latest` before pytest so these run in CI.

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

Still open (need server-side iteration):

- **deepseek-v4-flash baseline/multishot write broken code** (GridSearchCV errors →
  `submit_failed`); gemma baseline/multishot succeed (0.739 / 0.741). Prompt tuning.
- **Sandbox image name.** Default is `autovibe-gym-sandbox:latest`; the server only
  had `autovibe-gym:latest` built, and rootless Docker can't pull from docker.io.
  Server setup task: build the image under the expected tag.

## Blocked / Needs Decision

- Local Docker CLI is unavailable in this Windows workspace, so Docker kernel
  integration is verified in GitHub Actions rather than locally.
- Existing `GymEnv` remains for compatibility, but new iterative experiments
  should use `NotebookGymEnv`.
- Repository owner still needs to confirm the `main` ruleset requires the
  `Python tests` status check before merge if connector/API access cannot verify
  rulesets.

---

## Next Actions

1. [x] Все PR смержены в main
2. [x] TZ.md, PROTOCOL.md, EXPERIMENT_REPORT.md синхронизированы
3. [ ] Запустить `python -m experiments.run_matrix --mode local` на H200 → получить notebook-era experiment results
4. [ ] Обновить EXPERIMENT_REPORT.md с новыми результатами после п.3

---

## Changelog

| Date | Change |
|------|--------|
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

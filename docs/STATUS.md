# AutoVibe Gym - Live Status

**Last updated:** 2026-06-02 (dashboard: full Dataset Center, dataset creation wizard, raw/prepared config workflow)
**Phase:** Hardening after first full H200 recon + building the local control-panel dashboard for configuring/launching/inspecting runs.

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
| `run_baseline.py` | Done | single-shot control preserved; prompts require raw-DataFrame pipelines; missing score is not logged as zero |
| `run_multishot.py` | Done | logged as `repeated_single_shot`; prompts require raw-DataFrame pipelines; not the fair checklist control |
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

Result after dashboard hardening: `198 passed`, coverage `74.70%`.
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
- **Verified after hardening:** backend API smoke confirms `/api/health`,
  `/api/runs`, and `/api/runs/{id}/checklist` agree on `11/12` and `0.88` for a
  legacy MLflow run without episode events; browser smoke on desktop/mobile
  confirms `11/12`, `88%`, no stale `92%`, no console errors, and no page-level
  horizontal overflow.
- **Run:** `dashboard/server/run.sh` (API :8000) + `cd dashboard/web && npm i && npm run dev` (:5173).

## Blocked / Needs Decision

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

1. [x] Все PR смержены в main
2. [x] TZ.md, PROTOCOL.md, EXPERIMENT_REPORT.md синхронизированы
3. [ ] Запустить `python -m experiments.run_matrix --mode local` на H200 → получить notebook-era experiment results и подтвердить fixed single-shot/repeated multishot сабмиты
4. [ ] Обновить EXPERIMENT_REPORT.md с новыми результатами после п.3

---

## Changelog

| Date | Change |
|------|--------|
| 2026-06-02 | Dataset Center full workflow: backend staged uploads/URL downloads/safe archive extraction/table preview/create-from-config/config editing, React Dataset Center search/filter/sort, full creation wizard, seven-tab detail page, docs and backend tests |
| 2026-06-02 | Single-shot/repeated now show code + checklist coverage in the dashboard: the legacy runners emit a synthesized episode (solution.ipynb, notebook_events, feedback_trace, summary) into `--workspace-dir` and log `checklist_coverage` measured from the generated code, so Notebook/Trajectory/Checklist tabs populate for these modes too |
| 2026-06-02 | Single-shot/repeated now produce a score locally: raised legacy executor timeout 60→300s, prompts require a fitted predict-ready model, and the runner auto-fits an unfitted submitted model before scoring (gpt-oss-120b left it unfitted). Verified single_shot=0.931 f1 on example_dry_bean |
| 2026-06-02 | Dashboard local-execution fix: single-shot/repeated use the legacy executor which defaulted to `docker` from .env → "no candidate" on a Mac without Docker. Local launches now force the in-process `subprocess` executor + local kernel (env `AUTOVIBE_DASHBOARD_EXECUTOR`) |
| 2026-06-02 | Dashboard visual polish: fixed sidebar (position:fixed), dumbbell logo replacing the «A» mark, cleaner gear/trash icons, and a rebuilt trajectory timeline — per-step icon badges by step kind (add/edit/delete/restart/run/validate/submit) with opaque fills and a clean connector line |
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

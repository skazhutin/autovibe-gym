# AutoVibe Gym - Live Status

**Last updated:** 2026-06-02 (dashboard execution modes + config-driven dataset ingestion verified end-to-end)
**Phase:** Extending the local control-panel dashboard from run inspection/launch into full dataset configuration/preparation while keeping the notebook + Docker stack green.

---

## Current Sprint Goal

Ship config-driven dataset ingestion across CLI + dashboard, including richer
dataset context for agents, safe multi-format readers, and raw/pre-split
preparation flows without regressing the notebook/Docker environment.

---

## Status by Component

### Core Gym (`gym/`)

| File | Status | Notes |
|------|--------|-------|
| `notebook.py` | Done | nbformat v4 document editing, stable cell ids, revisions, outputs, Python export |
| `jupyter_kernel.py` | Done | persistent local `ipykernel`; Docker kernel backend with loopback-only ZMQ ports and workspace path translation |
| `notebook_env.py` | Done | real notebook action loop, clean restart-and-run-all, validate, submit, public/private artifacts, and dataset-context injection into prompts |
| `datasets.py` | Done | metadata loader now carries `dataset_notes` and formats safe dataset context blocks for agent-facing prompts |
| `dataset_ingestion.py` | Done | shared YAML-backed ingestion pipeline for raw/pre_split/multi-table datasets, validation, preview, URL download, and preparation |
| `tabular_io.py` | Done | safe tabular readers for CSV/TSV/TXT, Excel, JSON/JSONL, Parquet/Feather/ORC, and archive members |
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
| `run_gym.py` | Done | uses `NotebookGymEnv`, logs notebook/process/private metrics, artifacts to MLflow, and passes structured dataset context into the agent prompt |
| `run_baseline.py` | Done | single-shot control preserved; prompts require raw-DataFrame pipelines, include dataset context, and missing score is not logged as zero |
| `run_multishot.py` | Done | logged as `repeated_single_shot`; prompts require raw-DataFrame pipelines, include dataset context, and are not the fair checklist control |
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
| Dataset ingestion / preparation pipeline | Passing |
| Dashboard dataset API flow | Passing |
| Docker kernel integration | Passing locally after sandbox image build; also runs in GitHub Actions when Docker is available |
| Step-budget semantics | Passing |

---

## Current Verification

Latest local verification:

```bash
python3 -m pytest tests/test_dataset_pipeline.py -q
cd dashboard/web && npm run build
python3 -m pytest -q
```

Results:

- `tests/test_dataset_pipeline.py`: `37 passed`
- `dashboard/web` production build: passed
- full suite: `209 passed, 1 warning in 252.96s`

The Docker-backed notebook integration now passes locally against a freshly
built `autovibe-gym-sandbox:latest` image. GitHub Actions remains the source of
truth for the Linux sandbox path.

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

- **Sandbox image name.** Default is `autovibe-gym-sandbox:latest`; local
  verification passed after building that tag explicitly. Any remote host that
  only has `autovibe-gym:latest` still needs the expected tag built locally.

## Web Dashboard (`dashboard/`, branch `dev/claude/web-dashboard`)

Local control panel, separate from `gym/`. Reuses the project `.venv`.

- **Backend** (`dashboard/server`, FastAPI): reads runs from MLflow and parses
  episode artifacts into notebook/trajectory/checklist/errors/logs (per-item
  checklist closure reconstructed by replaying public notebook events through the
  gym's own `NotebookChecklist`); models JSON registry seeded from `.env` with
  OpenAI-compatible health probe; run launcher spawning
  `run_baseline/run_multishot/run_gym` subprocesses (shared MLflow store), live
  status + process-log tail + stop; datasets now reuse the shared ingestion
  layer for config/file/URL management, source preview, validation, and
  preparation while keeping the legacy prepared-upload path compatible.
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
- **Dataset UX** (`dashboard/web`): the datasets list now creates empty dataset
  configs first, then opens a full editor with config/preview/validate/prepared
  tabs; supports raw vs pre-split ingestion, multi-table joins, per-file format
  and read options, uploads, URL downloads, dataset notes/LLM context, and a
  proportion bar for split ratios that matches the existing dashboard design
  language.
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
- **Verified after hardening:** backend API smoke confirms `/api/health`,
  `/api/runs`, and `/api/runs/{id}/checklist` agree on `11/12` and `0.88` for a
  legacy MLflow run without episode events; browser smoke on desktop/mobile
  confirms `11/12`, `88%`, no stale `92%`, no console errors, and no page-level
  horizontal overflow.
- **Verified for dataset ingestion:** API tests cover create, upload, save,
  preview, validate, and prepare flows; the frontend build passes with the new
  dataset editor, split-visualization UI, richer preview diagnostics, and
  prepared meta summary.
- **Run:** `dashboard/server/run.sh` (API :8000) + `cd dashboard/web && npm install && npm run dev` (:5173).

## Blocked / Needs Decision

- Local Docker CLI is available in this workspace; the Docker-backed notebook
  integration test now passes locally with the expected sandbox image tag.
  GitHub Actions still verifies the Linux sandbox image path.
- Existing `GymEnv` remains for compatibility, but new iterative experiments
  should use `NotebookGymEnv`.
- Repository owner still needs to confirm the `main` ruleset requires the
  `Python tests` status check before merge if connector/API access cannot verify
  rulesets.

---

## Next Actions

1. [x] Dashboard datasets moved from CSV-only upload to shared config-driven ingestion
2. [x] Local verification now includes a passing Docker-backed full test suite (`209 passed`)
3. [ ] Run dashboard smoke on a real multi-table dataset and a URL-ingested dataset
4. [ ] Open/land the dataset-ingestion PR after manual dashboard verification and CI
5. [ ] Запустить `python -m experiments.run_matrix --mode local` на H200 и обновить `EXPERIMENT_REPORT.md`

---

## Changelog

| Date | Change |
|------|--------|
| 2026-06-02 | Dashboard checklist consistency: tab count uses the recorded `checklist_coverage` (single source of truth, matches the run banner) and exactly that many items render green; aligned the live-banner count to the same formula |
| 2026-06-02 | Dashboard execution modes: per-run selector «на сервере (SSH) / на компьютере» on New Run (overrides the global default); local mode runs gym on the machine and calls the remote LLM (works off-VPN) |
| 2026-06-02 | Dashboard remote execution: run the gym on the GPU server over SSH while the site stays local (`services/remote_exec.py`: ssh/rsync launch + artifact sync + run-summary parse; key auth or optional expect password); configurable in Settings with a connectivity probe |
| 2026-06-02 | Dashboard single-app server mode: FastAPI serves the built SPA (one process) so the whole dashboard can run on the server; added `serve.sh` and deploy docs |
| 2026-06-02 | Dashboard live updates: launches write to a known workspace dir and the backend reads in-flight artifacts, so step/checklist/notebook/trajectory/logs advance during a run (2.5s polling); models registry seeded with team gemma/deepseek; header pill switched to LLM "Сервер онлайн/офлайн" |
| 2026-06-02 | Added shared dataset ingestion (`gym/dataset_ingestion.py`, `gym/tabular_io.py`), passed dataset context into agent prompts, rebuilt the dashboard dataset flow around config/preview/validate/prepare, added matching split visualization in raw mode plus preview diagnostics/meta summary, and verified the full suite locally with the sandbox image (`209 passed`) |
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

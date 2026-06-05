# AutoVibe Gym - Live Status

**Last updated:** 2026-06-05 (full visibility for failed runs across all modes)
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

Last local run:

```bash
python -m pytest
```

Result after all-mode orchestration: `222 passed` (one sklearn
`InconsistentVersionWarning` from the Docker notebook pickle smoke).
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
| 2026-06-05 | Failed runs now expose all available info across every mode: repeated single-shot writes a full multi-attempt episode (every attempt's code + error visible in Notebook/Trajectory/Errors/Logs, not just the best), the run record carries `failReason`/`finalStatus` from the runner's `null_reason`/`final_status`, and the detail page always shows the fail banner with the status label |
| 2026-06-05 | Dashboard launcher clamps `--max-tokens` to the selected model's `maxTokens` cap, so providers with tight per-minute token limits (e.g. Groq free ~6000 TPM) don't 413 when the New Run form leaves the default high |
| 2026-06-05 | Dashboard tabs visual fix: removed the 1px vertical overflow in tab menus by moving the divider into the tab strip and dropping the negative tab margin |
| 2026-06-04 | Dashboard logo refresh: replaced the H-like yellow-square mark with a diagonal dumbbell AutoVibe mark |
| 2026-06-04 | Dashboard sidebar cleanup: removed the bottom `Команда / локальный режим` block and its unused styles |
| 2026-06-04 | Normalized product-mode display labels: dashboard short labels now show `Flexible gym`, and fixed transitions display as `Fixed gym` in dashboard and shared matrix metadata |
| 2026-06-02 | Dashboard trajectory visual fix: aligned the live "агент выполняет шаг…" spinner with the timeline marker column and adjusted the connector so the grey line meets the spinner's top center |
| 2026-06-02 | Agent thoughts/scratchpad: optional `notes` field on any action; with `--enable-thoughts` (gym/iterative) the env stores notes, re-injects them every turn ([YOUR NOTES SO FAR]), persists `scratchpad.json`. Dashboard: New Run toggle (gym/iterative) + a «Мысли» tab rendering the notes timeline. One PR. |
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

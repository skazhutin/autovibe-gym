# AGENT NOTES — web dashboard (private working memory)

> My scratch/context file so I don't re-derive the backend each session. Not a team doc.
> Branch: `dev/claude/web-dashboard`. Folder: `dashboard/` (separate from `gym/`).

## Goal
Local web dashboard to run LLM experiments on AutoVibe Gym: pick model + run-mode +
dataset, configure budget, launch, watch live, read the LLM's notebook/trajectory,
see score+errors, manage datasets/models, compare runs. Design = `~/Desktop/design`
(hi-fi React/Babel prototype + README with tokens). UI in Russian. Colors: T-Bank
yellow `#FFDD2D` + dark `#333333`.

## Stack decision
- Backend: **FastAPI** (project `.venv` already has fastapi 0.136, uvicorn 0.48,
  mlflow 3.12, pandas 2.3, pydantic 2.13 — Python 3.11). Run with `.venv/bin/python`.
- Frontend: **Vite + React + TS** (node v25, npm 11 available). Plain CSS vars for
  tokens (no Tailwind, to match prototype's CSS-variable system 1:1).
- Dev: vite proxies `/api` → FastAPI `:8000`.

## Backend reality (what to wrap)
- Experiments = python processes logging to **MLflow** (`mlflow.db` + `mlruns/` at repo root).
  Runner CLIs: `experiments/run_gym.py` (notebook modes), `run_baseline.py` (single_shot),
  `run_multishot.py` (repeated_single_shot), `run_fixed.py`, `run_matrix.py`.
- `run_gym.py` args: `--dataset-dir <dir>` OR `--dataset <csv> --target <col>`, `--mode {local,cloud}`,
  `--model`, `--max-steps`, `--max-tokens`, `--sandbox-timeout`,
  `--episode-mode {gym_with_checklist,iterative_no_checklist}`, `--experiment-name`, `--run-name`,
  `--workspace-dir`, `--seed`.
- MODE_DEFAULTS: local={max_steps30,max_tokens8192,timeout120}, cloud={20,4096,120}.
- Episode modes (gym/modes.py): only 2 notebook modes. single_shot + repeated_single_shot
  come from baseline/multishot runners. So **4 UI modes → 3 runner scripts**:
  - single → run_baseline.py
  - repeated → run_multishot.py (logged as `repeated_single_shot`)
  - iterative → run_gym.py --episode-mode iterative_no_checklist
  - gym → run_gym.py --episode-mode gym_with_checklist
- MLflow run params logged (run_gym): mode, episode_mode/experiment_type, model, dataset,
  protocol_version, max_steps, max_tokens, sandbox_timeout, dataset_suite/split/role/sampled...
- MLflow metrics: checklist_coverage, private_checklist_coverage, steps_used, error_count,
  has_test_metric, valid_submit, submit_failed, input_tokens, output_tokens, elapsed_seconds,
  notebook_cells_final, ..., best_validation_metric, final_test_metric / test_metric.
- Artifacts logged to MLflow under `episode/` (public) and `episode_private/` (private):
  - public: `solution.ipynb`, `validation_trajectory.json`, `episode_summary.json`, events json.
  - private: trajectory_private.json, episode_summary.json (has final_test_metric).
- `agent.run()` summary keys: + input_tokens, output_tokens, model, forced_submit/stopped_reason.
- env.get_summary() keys incl: steps_used, checklist_coverage, valid_submit, test_metric,
  final_test_metric, submit_failure_type, episode_workspace, private_episode_dir, + more.

## Datasets
- `datasets/<name>/prepared/{train,val,test}.csv + meta.json` is the runnable form.
  Some only have `raw_data/ + config.yaml` (need `scripts/prepare_datasets.py`).
- meta.json (DatasetMetadata.from_dict): `target_col`/`target` (req), `metric_name`/`metric`,
  `task_type`, `source`, `suite`, `split_strategy`, `role`, `sampled`, `seed`(=42), `notes`.
- metric_from_name supports: f1_weighted, f1_macro, neg_rmse. infer: <=10 unique→f1_weighted else neg_rmse.
- Example datasets: example_student_dropout, example_room_occupancy, example_dry_bean,
  example_naticusdroid, example_phiusiil_phishing.

## Models / LLM config (.env)
LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL. gym/llm.py picks client (OpenAI-compat,
Gemini, LiteLLM). Cloud API `llm.letovo.site` exposes deepseek-v4-flash, gemma-4-26b (per STATUS).
Models registry → store as JSON config file in dashboard (server/data/models.json), since gym
has no model registry. health-check = cheap call via configured base url/key.

## Design tokens (from design/README.md) — see web/src/styles/tokens.css
accent #FFDD2D, accent-ink #8F7400, ink #333, text #262626, dim #6C6C6C, faint #A0A0A0,
bg #F5F5F5, surface #fff, surface-2 #F1F1F1, border #EAEAEA.
green #1F9D62/#E8F5EE/#C2E5D2, red #D14848/#FBEDED/#F1CBCB, orange #DD8A1E.
Dark: bg#262626 surface#333 surface-2#3E3E3E text#F0F0F0 dim#A6A6A6 border#474747 code#2B2B2B.
Fonts: Helvetica Neue UI + JetBrains Mono (Google Fonts). body 15px/1.5. H1 22/800 -.02em.
Radius lg=18 md=~12 mr=~10, pills 999. Shadows sm/md/lg as in README.
Sidebar 248px sticky 100vh. Content padding 28/32/64, max 1400, centered.
Syntax: comment #9aa0a6 italic, string #2E8B57, number #C76E00, keyword #9B4DCA,
function #3B6CC7, identifier #363636, operator #888.

## Screens (8) — detail in design/README.md "Screens"
dashboard, new (configurator), runs (history), detail (5 tabs: notebook/trajectory/
checklist/errors/logs — MOST IMPORTANT), compare, datasets(+detail+modals), models, settings.
Reference jsx files at ~/Desktop/design/reference/: data.jsx (entity shapes), ui.jsx (ICONS+
primitives), charts.jsx (CodeBlock + SVG charts), screens-*.jsx, app.jsx (shell+routing+live sim).

## Entity shapes (data.jsx) — API mirrors these
- Run: id, model, mode(single|repeated|iterative|gym), dataset, status(success|failed|null|running),
  score(nullable), baseline, checklist(0-8), errors, step, steps, tokIn, tokOut, startedMin,
  dur(sec), seed, temp, failReason?
- Model: id, name, provider(vLLM|OpenAI-совместимый|Gemini|LiteLLM), baseUrl, online, ctx
- Dataset: id, name, task(Регрессия|Классификация), metric, metricGoal(min|max), rows, cols,
  target, source, desc
- Checklist 8 items: id,label,desc,closedStep
- Notebook: cells [{n, code, out:{type:stdout|table|error|submit,...}}]
- Trajectory: steps [{step, action(code|validate|submit), title, code, feedback:[{ch:runtime|
  contract|checklist|checklist-hint|terminal, text}]}]
- Logs: [{role:system|user|assistant|tool, text}]
Score fmt: RMSE 1dp, RMSLE 4dp, else 3dp. Improve% over baseline: min→(b-s)/b*100 else (s-b)/b*100.

## API contract (planned) — prefix /api
GET  /health
GET/PUT /settings
GET  /datasets ; GET /datasets/{id} ; POST /datasets (upload) ; PUT /datasets/{id} ; DELETE
GET  /datasets/{id}/preview ; /datasets/{id}/columns
GET  /models ; POST /models ; PUT /models/{id} ; DELETE ; POST /models/{id}/health
GET  /runs ; GET /runs/{id} ; POST /runs (launch) ; POST /runs/{id}/stop
GET  /runs/{id}/notebook ; /trajectory ; /checklist ; /errors ; /logs
GET  /runs/{id}/events (SSE/poll for live)

## Progress log
- [done] branch + scaffolding
- [done] backend: datasets + models services/routers
- [done] backend: runs (mlflow_store + run_launcher) + router; smoke-tested vs real mlruns
- [done] frontend foundation: api client, theme, hooks, Icon, ui primitives, charts, CodeBlock, Layout shell, routing
- [done] screens: Dashboard, Runs, NewRun, RunDetail(5 tabs), Compare, Datasets+Detail+modals, Models, Settings
- [done] verify: build clean; uvicorn HTTP ok; vite proxy ok; launcher cmd ok
- [done] docs/STATUS.md updated
- gotchas: python-multipart needed for uploads (installed in venv). data/ gitignored (has API keys).
  checklist tab coverage = replay (self-consistent) vs header chip uses MLflow metric (can differ slightly).
  LIVE UPDATES (done): launcher passes --workspace-dir data/runs/<id>/workspace; gym's
  _record_observation already calls _save_artifacts() per step → artifacts flush live there.
  mlflow_store parsing is now dir-based (episode_dir arg, no lru_cache); runs router _episode_dir()
  uses live workspace while running else mlflow .../artifacts/episode; _enrich_live() folds
  episode_progress (step/checklist/errors via replay) into running runs (list + detail).
  Frontend polls detail+tabs every 2.5s. Tokens still only at end (in agent, logged to mlflow).
  EXECUTION LOCALITY: launcher spawns LOCAL .venv python, cwd=local repo; kernel backend=local
  → gym + notebook model-training run on the MACHINE RUNNING THE BACKEND. Only LLM is remote.
  So to offload the laptop → run the whole dashboard ON the server. FastAPI now serves the built
  SPA (main.py: mounts /assets + SPA catch-all when web/dist exists) → one process. serve.sh
  (HOST=0.0.0.0 PORT=8011 BUILD=1) for server. Deploy steps in dashboard/README.md. Server venv
  needs `pip install -r dashboard/server/requirements.txt` (fastapi/uvicorn/multipart) + current gym.
  REMOTE-EXEC (SSH) mode (done): site stays local, gym runs on server over SSH. services/remote_exec.py
  (ssh/rsync, key-auth default + optional password via expect). run_launcher branches in launch() when
  remote_exec.is_enabled(); _launch_remote nohups runner on server (workspace+log under remote_runs_dir),
  _refresh_remote rsyncs remote workspace→data/runs/<id>/ (throttled 2.5s) + checks pid alive; on finish
  parses run_gym's stdout "=== Run Summary ===" JSON (final_test_metric/valid_submit/...) for the result
  (no server MLflow needed). dataset passed as relative datasets/<id>. Config in Settings screen
  (remote_* keys in settings.json, password masked in GET, never wiped by masked PUT). POST
  /api/settings/remote-check probes ssh+repo+gym. NOTE: couldn't test against real server — 10.8.52.11
  unreachable from agent sandbox (only user's Mac has VPN). Needs user testing + ssh key.
  EXECUTION SELECTOR (done): New Run has «на сервере (SSH) / на компьютере»; LaunchPayload.execution
  overrides global toggle; run_launcher.launch computes want_remote; remote_exec.is_configured() gates it.
  MODELS: team gemma/deepseek seed at llm.letovo.site (INTERNAL host — only on VPN; off-VPN → ConnectTimeout).
  Cerebras gpt-oss-120b added LOCALLY to data/models.json (gitignored, key not in git) as a public option.
  CHECKLIST CONSISTENCY (done, branch dev/claude/checklist-coverage-fix): tab coverage/count = recorded
  checklist_coverage metric (== banner); episode_progress uses same round(coverage*total) formula; and
  exactly `closed` items render green (replay-detected first, then canonical order) so ticks == number.
  DOCS RULE: keep docs/STATUS.md changelog + this file updated every change (user asked). main is protected;
  fixes go on dev/claude/* branches via PR.
- next ideas: open draft PR; optional gym-side incremental artifact flush for true live streaming;
  run a real end-to-end launch once an LLM endpoint is reachable.
</content>
</invoke>

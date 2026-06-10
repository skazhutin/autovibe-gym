# AutoVibe Gym

AutoVibe Gym is a research environment for testing whether an LLM solves
tabular ML tasks better when it can work iteratively inside a controlled
notebook-like gym.

The project compares plain one-shot code generation against multi-step agent
runs that receive runtime feedback, contract checks, and optional data-science
checklist hints. Hidden test data stays private until the final submit.

## Why This Exists

The main research question is not "can an LLM get a good score once?" The goal is
to measure deltas between interaction modes:

```text
delta = score(iterative mode) - score(single-shot baseline)
```

All reported metrics are normalized so higher is better. For regression tasks
this means metrics such as `neg_rmse`, where `-0.48` is better than `-0.58`.

AutoVibe Gym records more than a final score:

- final hidden-test metric, when a valid submit exists
- checklist coverage
- execution errors and preflight failures
- token usage and steps used
- public notebook trajectory
- private evaluator artifacts that are never shown to the agent

## Current Product Modes

The shared runner supports five product modes:

| Mode | CLI key | Description |
| --- | --- | --- |
| Single-shot | `single_shot` | One LLM response, no iterative feedback |
| Repeated single-shot | `repeated_single_shot` | Several independent one-shot attempts |
| Free gym | `free_gym` | Real notebook loop with runtime/contract feedback |
| Directive gym | `directive_gym` | Free gym plus generic checklist hints |
| Fixed gym | `fixed_gym` | Multi-stage guided gym run |

Use `experiments.run --mode all` to launch all five modes for one dataset/model,
or `--modes ...` to launch a selected batch.

## Repository Layout

```text
gym/                 core environment, protocol, Jupyter backend, LLM clients
experiments/         CLI runners and comparison utilities
dashboard/           FastAPI + React control panel
datasets/            prepared/example datasets and local dataset workspaces
docs/                project notes, protocol, status, reports
tests/               unit and regression tests
Dockerfile.sandbox   sandbox image for legacy single-shot code execution
```

Generated data, local model keys, MLflow outputs, run workspaces, and dashboard
state are intentionally gitignored.

## Quickstart: Local Python

Requirements:

- Python 3.10+
- Node.js 18+ for the dashboard frontend
- Docker, if you want the Docker sandbox/backend
- an OpenAI-compatible, Gemini, or LiteLLM-accessible model

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r dashboard/server/requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r dashboard/server/requirements.txt
```

## Configure Models

Models live in the shared registry at `dashboard/server/data/models.json`.
That file is local state and should not be committed.

List configured models:

```bash
python -m experiments.models list
```

Add an OpenAI-compatible model:

```bash
python -m experiments.models add \
  --name my-model-name \
  --provider OpenAI-compatible \
  --base-url http://localhost:8000/v1 \
  --api-key local-or-secret-token
```

Add a Gemini model:

```bash
python -m experiments.models add \
  --name gemini-2.5-flash \
  --provider Gemini \
  --api-key your-token
```

Runners accept either a model id or a model name from the registry.

## Run The Dashboard

Start the backend API:

```bash
dashboard/server/run.sh
```

On Windows, run the same backend directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn dashboard.server.app.main:app --port 8000
```

Start the frontend:

```bash
cd dashboard/web
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

The dashboard can:

- manage dataset tasks and prepared splits
- manage the model registry
- launch local or SSH-remote runs
- stream live run progress
- inspect notebooks, trajectories, checklist evidence, logs, and failures
- compare finished runs

For a single-process server deployment, build and serve the dashboard with:

```bash
BUILD=1 PORT=8011 HOST=0.0.0.0 dashboard/server/serve.sh
```

## Run Experiments From CLI

Run all product modes for one dataset/model:

```bash
python -m experiments.run \
  --dataset-dir datasets/example_dry_bean/prepared \
  --mode all \
  --model my-model-name
```

Run selected modes:

```bash
python -m experiments.run \
  --dataset-dir datasets/example_dry_bean/prepared \
  --modes single_shot directive_gym fixed_gym \
  --model my-model-name
```

Run individual modes:

```bash
python -m experiments.run_baseline --dataset-dir datasets/example_dry_bean/prepared --model my-model-name
python -m experiments.run_multishot --dataset-dir datasets/example_dry_bean/prepared --model my-model-name
python -m experiments.run_gym --dataset-dir datasets/example_dry_bean/prepared --episode-mode free_gym --model my-model-name
python -m experiments.run_gym --dataset-dir datasets/example_dry_bean/prepared --episode-mode directive_gym --model my-model-name
python -m experiments.run_fixed --dataset-dir datasets/example_dry_bean/prepared --model my-model-name
```

Compare finished runs:

```bash
python -m experiments.compare
```

## Dataset Layout

Preferred layout:

```text
datasets/<dataset_name>/
  prepared/
    train.csv
    val.csv
    test.csv
    meta.json
```

Minimal `meta.json`:

```json
{
  "name": "example_dry_bean",
  "target_col": "Class",
  "metric_name": "f1_macro",
  "task_type": "classification",
  "seed": 42
}
```

Regression tasks can use `neg_rmse`. Since higher is always better, the stored
score is the negative RMSE.

Prepare configured example datasets:

```bash
python scripts/prepare_datasets.py --list
python scripts/prepare_datasets.py --dataset example_dry_bean
```

## Agent Protocol

Interactive gym modes use canonical JSON actions. Every action has a `type` and
a deterministic `stage`.

Examples:

```json
{"type": "think", "stage": "planning", "thoughts": "I will inspect the schema, build a raw-input pipeline, validate it, and submit only when ready."}
```

```json
{"type": "add_cell", "stage": "data_schema_inspection", "cell_type": "code", "source": "print(train_df.shape); print(train_df.dtypes)", "execute": true}
```

```json
{"type": "restart_and_run_all", "stage": "reproducibility_check"}
```

```json
{"type": "validate", "stage": "validation_analysis", "model_var": "model"}
```

```json
{"type": "submit", "stage": "submission", "model_var": "model"}
```

`test_df` is never injected into the agent workspace. A successful submit only
tells the agent that the candidate was accepted; the hidden score is stored in
private artifacts and MLflow metrics.

## Sandbox And Privacy Boundary

Notebook gym modes use a persistent Jupyter kernel and write a real
`solution.ipynb`. The local Jupyter backend is useful for research and demos,
but it is not a full OS sandbox for untrusted code.

For stronger notebook isolation, set:

```bash
AUTOVIBE_KERNEL_BACKEND=docker
```

Legacy single-shot and repeated-single-shot runners use `CodeExecutor`. Build
the sandbox image with:

```bash
docker build -f Dockerfile.sandbox -t autovibe-gym-sandbox:latest .
```

Useful executor settings:

```bash
AUTOVIBE_EXECUTOR_BACKEND=docker
AUTOVIBE_SANDBOX_IMAGE=autovibe-gym-sandbox:latest
AUTOVIBE_SANDBOX_THREADS=1
```

The immediate privacy guarantee is physical hidden-test isolation: hidden test
files, labels, hidden score, evaluator diagnostics, and candidate pickle paths
are not exposed to agent-facing artifacts.

## MLflow

By default runners use local SQLite MLflow tracking:

```text
sqlite:///mlflow.db
```

Open a local MLflow UI:

```bash
python -m mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

If the installed MLflow version changes and `/api/runs` starts failing with an
out-of-date schema error, back up the database and migrate it:

```bash
cp mlflow.db mlflow.db.bak
python -m mlflow db upgrade sqlite:///mlflow.db
```

## Development Checks

Run the focused test suite:

```bash
python -m pytest
```

Dashboard frontend checks:

```bash
cd dashboard/web
npm run build
```

Before opening a PR:

```bash
git status --short
git diff --check
python -m pytest
```

## What Not To Commit

Do not commit:

- `.env` or API keys
- local `dashboard/server/data/` state
- `mlflow.db`, `mlruns/`, `outputs/`, `logs/`, run workspaces
- private/raw datasets unless explicitly curated for the repo
- notebook checkpoints, caches, or local agent memory

See `docs/GIT_WORKFLOW.md` and `docs/STATUS.md` before preparing a PR.

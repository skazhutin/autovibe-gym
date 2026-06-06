# AutoVibe Gym

AutoVibe Gym is an iterative AutoML environment where an LLM writes ML code,
receives structured feedback, improves the solution, and submits one final model
against a hidden test split.

The goal is not the highest score — it is a **verifiable evaluation environment**
that makes agent failures observable and private test evaluation tamper-proof.

---

## Quickstart — run all 4 experiment modes in 5 minutes

### Prerequisites

- Docker installed
- Git
- An API key for the LLM (OpenAI-compatible endpoint)

### Step 1 — Clone and build

```bash
git clone https://github.com/skazhutin/autovibe-gym.git
cd autovibe-gym
docker build -t autovibe-gym .
```

### Step 2 — Download raw datasets

The five example datasets are configured but raw files must be downloaded once.
The simplest dataset that works immediately is `student_dropout` (CSV, no account needed):

```bash
# Download student_dropout raw data
mkdir -p datasets/student_dropout/raw_data
curl -L "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip" \
  -o datasets/student_dropout/raw_data/predict+students+dropout+and+academic+success.zip
```

Or copy the `datasets/student_dropout/prepared/` directory directly from someone who already has it.

### Step 3 — Prepare splits

```bash
docker run --rm \
  -v "$(pwd):/autovibe" \
  -e MLFLOW_TRACKING_URI=file:///autovibe/mlruns \
  autovibe-gym \
  -m scripts.prepare_datasets --dataset student_dropout
```

### Step 4 — Configure a model

```bash
python -m experiments.models add \
  --name deepseek-v4-flash \
  --provider "OpenAI-совместимый" \
  --base-url "<provided separately>" \
  --api-key "<provided separately>"
```

### Step 5 — Run all 4 modes

```bash
DS="/autovibe/datasets/student_dropout/prepared"
DOCKER="docker run --rm \
  -v $(pwd):/autovibe \
  -e MLFLOW_TRACKING_URI=file:///autovibe/mlruns \
  autovibe-gym"

# Recommended: run the 4 product modes as one linked batch.
# This creates 4 separate MLflow runs with the same batch_id and
# requested_mode=all.
$DOCKER -m experiments.run \
  --dataset-dir $DS \
  --mode all \
  --budget-mode cloud \
  --model deepseek-v4-flash

# Or run each mode manually:

# Mode 1 — Single-shot (no feedback, ~15s)
$DOCKER -m experiments.run_baseline --dataset-dir $DS --mode cloud

# Mode 2 — Repeated single-shot (5 attempts, ~2 min)
$DOCKER -m experiments.run_multishot --dataset-dir $DS --mode cloud

# Mode 3 — Flexible transitions / gym (15 steps, ~5 min)
$DOCKER -m experiments.run_gym --dataset-dir $DS --mode cloud

# Mode 4 — Fixed transitions (5 stages, ~5 min)
$DOCKER -m experiments.run_fixed --dataset-dir $DS --mode cloud
```

Or run the four product modes in one command with the batch runner:

```bash
$DOCKER -m experiments.run_all_modes_matrix \
  --datasets /autovibe/datasets/student_dropout/prepared \
  --models deepseek-v4-flash \
  --mode cloud
```

Use `experiments.run_all_modes_matrix` when you want the full
datasets × models × product modes matrix. Use `experiments.run --mode all`
for the normal one-dataset launch.

### Step 6 — Compare results

```bash
docker run --rm \
  -v "$(pwd):/autovibe" \
  -e MLFLOW_TRACKING_URI=file:///autovibe/mlruns \
  autovibe-gym \
  -m experiments.compare --dataset student_dropout
```

Expected output (values will vary by model):

```
requested_mode  batch_id          mode_label            test_metric  steps  tokens   elapsed
--------------  ----------------  --------------------  -----------  -----  -------  -------
all             20260603-...      single_shot                 0.747      1    2 120      12s
all             20260603-...      repeated_single_shot        0.730      5   11 228     110s
all             20260603-...      gym_with_checklist          0.745     14  146 013     111s
all             20260603-...      fixed_transitions            null     17  218 322     218s
```

> The single-shot baseline wins on score **and** cost.
> The gym reveals what the agent did (checklist coverage, failure types).
> Fixed transitions with rigid stage order performs worst when the agent
> exhausts its preprocessing budget before finding good features.
> `null` means no valid hidden-test submission, not a score of zero.

---

## Key finding

More interaction ≠ better score. The environment's value is **diagnostic**:
it shows *why* an agent succeeded or failed, not just the final number.
This is invisible to a plain leaderboard.

---

## Core Loop

```text
GymAgent
  -> LLMClient
  -> JSON Action
  -> NotebookGymEnv.step()
  -> real .ipynb document + persistent Jupyter kernel
  -> runtime / contract / checklist feedback
  -> Observation feedback
  -> next JSON Action
```

Actions always use canonical `type` and must include a deterministic `stage`:

```json
{"type": "code", "stage": "data_schema_inspection", "code": "print(train_df.shape)"}
```

Legacy `code` actions are still accepted, but the current protocol is
cell-oriented:

```json
{"type": "add_cell", "stage": "feature_pipeline_building", "cell_type": "code", "source": "print(train_df.shape)", "execute": true}
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

When `--enable-thoughts` is set, every action also includes a short visible
`thoughts` summary. The first action must be a non-mutating planning thought:

```json
{"type": "think", "stage": "planning", "thoughts": "I will inspect the data, build and validate a reproducible pipeline, and submit only when ready."}
```

Useful gym tools are also JSON actions:

```json
{"type": "inspect_data", "stage": "data_schema_inspection"}
{"type": "profile_data", "stage": "data_quality_inspection", "profile": "compact"}
{"type": "list_candidates", "stage": "candidate_training"}
{"type": "check_candidate", "stage": "validation_analysis", "model_var": "auto"}
{"type": "quick_validate", "stage": "validation_analysis", "model_var": "auto"}
{"type": "finalize", "stage": "submission", "model_var": "auto"}
```

Optional diagnostics are available when configured/installed:

```json
{"type": "profile_data", "stage": "data_quality_inspection", "profile": "ydata"}
{"type": "cleanlab_diagnose", "stage": "validation_analysis", "model_var": "auto"}
{"type": "tune_hyperparameters", "stage": "model_improvement", "model_var": "model", "search_space": {}, "n_trials": 10}
```

Each episode creates `solution.ipynb` and treats it as the source of truth.
The LLM can add, update, delete, move, run, and inspect real notebook cells.
The kernel is persistent during interactive work, so variables survive between
executions. Final acceptance still requires a clean `restart_and_run_all`,
environment-controlled `validate`, and then `submit`.

The kernel contains `train_df`, `val_df`, `target_col`, `pd`, and `np`.
`test_df` and hidden split files are never copied into the episode workspace.
Successful submit returns only:

```text
[SUBMITTED] Final candidate accepted. Episode finished.
```

The hidden test metric is stored only in private summaries and MLflow metrics.

Agent-visible episode artifacts are saved in the episode workspace:

```text
final_notebook.ipynb
final_notebook.py
notebook_events.json
feedback_trace.json
validation_trajectory.json
episode_summary.json
scratchpad.json          # only when thoughts mode is enabled
```

These public JSON artifacts are sanitized: hidden test metrics, private
checklist coverage, submit failure types, and candidate pickle paths are not
written where the kernel can read them. Private evaluator artifacts live in a
separate private episode directory that is not mounted into the Docker kernel:

```text
episode_summary.json
feedback_trace_private.json
notebook_events_private.json
validation_trajectory_private.json
agent_trace_private.jsonl
cell_executions_private.jsonl
candidate_diagnostics_private.jsonl
data_profile_private.json
artifacts/*.pkl
```

## Execution Sandbox

The default iterative Gym runner uses a local real Jupyter kernel. This provides
real notebook behavior, saved outputs, rich displays, and clean replay. The
implementation strips common secret environment variables from the kernel and
physically keeps hidden test files and private evaluator artifacts outside the
episode workspace.

`gym.jupyter_kernel` defines `KernelExecutionBackend`,
`LocalJupyterKernelBackend`, and `ContainerJupyterKernelBackend`. Set
`AUTOVIBE_KERNEL_BACKEND=docker` to run each notebook kernel in a Docker
container with an internal network, read-only root filesystem, dropped
capabilities, no-new-privileges, resource caps, `/tmp` tmpfs, and ZMQ ports
published only on `127.0.0.1`.

The legacy `CodeExecutor` is still used by single-shot/repeated-single-shot
baselines and can run through Docker by default:

Build the sandbox image once:

```bash
docker build -f Dockerfile.sandbox -t autovibe-gym-sandbox:latest .
```

Default sandbox settings:

```bash
AUTOVIBE_EXECUTOR_BACKEND=docker
AUTOVIBE_SANDBOX_IMAGE=autovibe-gym-sandbox:latest
AUTOVIBE_SANDBOX_MEMORY_MB=2048
AUTOVIBE_SANDBOX_CPUS=1
AUTOVIBE_SANDBOX_PIDS_LIMIT=128
```

Unit tests can use the lightweight subprocess fallback:

```bash
AUTOVIBE_EXECUTOR_BACKEND=subprocess python -m pytest
```

## LLM Providers

AutoVibe Gym uses the shared model registry for model names, providers,
endpoints, and per-model API keys. Use the dashboard Models page or the CLI:

```bash
python -m experiments.models list
python -m experiments.models add \
  --name Qwen/Qwen2.5-Coder-7B-Instruct \
  --provider "OpenAI-совместимый" \
  --base-url http://localhost:8000/v1 \
  --api-key local
```

OpenAI-compatible entries cover local vLLM, OpenAI, LiteLLM proxies, Yandex AI
through its OpenAI-compatible endpoint, and any other `/v1/chat/completions`
server. Gemini does not use a Base URL:

```bash
python -m experiments.models add \
  --name gemini-2.5-flash \
  --provider Gemini \
  --api-key your-token
```

Then pass either the model id or the model name to any runner:

```bash
python -m experiments.run_gym --dataset-dir datasets/demo/prepared --model gemini-2.5-flash
```

## MLflow

Experiment runners and `experiments.compare` use the same tracking default:
local `sqlite:///mlflow.db`. `MLFLOW_TRACKING_URI` is only needed when you want
to point at a separate MLflow server.

To open the local UI:

```bash
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

## Experiment Modes

`experiments.run_gym` supports two fair iterative modes on the same Jupyter
backend:

- `iterative_no_checklist`: real notebook, runtime feedback, contract feedback,
  no data-science checklist hints.
- `gym_with_checklist`: the same notebook/backend/budget/actions plus selective
  generic checklist hints.

The non-notebook controls remain:

- `single_shot`: one solution without the interactive notebook loop
  (`experiments.run_baseline`).
- `repeated_single_shot`: repeated attempts with execution feedback
  (`experiments.run_multishot`).

For a single entrypoint, use `experiments.run`:

```bash
python -m experiments.run --dataset-dir datasets/example_dry_bean/prepared --mode single_shot --model deepseek-v4-flash
python -m experiments.run --dataset-dir datasets/example_dry_bean/prepared --mode all --model deepseek-v4-flash --dry-run
```

`--mode all` expands to four separate product runs in this order:
`single_shot`, `repeated_single_shot`, `gym_with_checklist`,
`fixed_transitions`. Each child run logs `requested_mode`, `batch_id`,
`product_mode`, `mode_label`, and `mode_order` so `experiments.compare` can
group the batch without mixing metrics into one run.

Run the checklist ablation:

```bash
python -m experiments.run_gym --dataset-dir datasets/example_dry_bean/prepared --episode-mode iterative_no_checklist
python -m experiments.run_gym --dataset-dir datasets/example_dry_bean/prepared --episode-mode gym_with_checklist
```

## Dataset Layout

Legacy CSV mode is supported:

```bash
python3 -m experiments.run_gym --dataset datasets/wine_quality.csv --target quality
```

Preferred fixed split mode for experiments:

```text
datasets/<dataset_name>/
  train.csv
  val.csv
  test.csv
  meta.json
```

```bash
python3 -m experiments.run_gym --dataset-dir datasets/wine_quality
```

`meta.json` should include at least:

```json
{
  "name": "wine_quality",
  "target_col": "quality",
  "metric": "f1_weighted",
  "seed": 42
}
```

## Example datasets (config-driven)

Datasets can be organized as `datasets/<name>/{config.yaml,raw_data/,prepared/}`.
Run preparation with:

```bash
python scripts/prepare_datasets.py --list
python scripts/prepare_datasets.py --dataset example_dry_bean
python scripts/prepare_datasets.py --suite example_datasets
```

The repository commits only the five curated `datasets/example_*` datasets. Any
other dataset folders under `datasets/` are treated as local-only data and are
ignored by git.

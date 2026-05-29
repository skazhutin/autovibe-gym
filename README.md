# AutoVibe Gym

AutoVibe Gym is an iterative AutoML environment where an LLM writes ML code,
receives structured feedback, improves the solution, and submits one final model
against a hidden test split.

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

Actions:

```json
{"type": "code", "code": "print(train_df.shape)"}
```

Legacy `code` actions are still accepted, but the current protocol is
cell-oriented:

```json
{"type": "add_cell", "cell_type": "code", "source": "print(train_df.shape)", "execute": true}
```

```json
{"type": "restart_and_run_all"}
```

```json
{"type": "validate", "model_var": "model"}
```

```json
{"type": "submit", "model_var": "model"}
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

Episode artifacts are saved in the episode workspace:

```text
final_notebook.ipynb
final_notebook.py
notebook_events.json
feedback_trace.json
validation_trajectory.json
episode_summary.json
```

## Execution Sandbox

The current iterative Gym runner uses a local real Jupyter kernel. This provides
real notebook behavior, saved outputs, rich displays, and clean replay, but it
is not a full OS-level sandbox for untrusted code. The implementation strips
common secret environment variables from the kernel and physically keeps hidden
test files outside the episode workspace.

`gym.jupyter_kernel` defines `KernelExecutionBackend`,
`LocalJupyterKernelBackend`, and a future `ContainerJupyterKernelBackend` hook.
Full container isolation for the Jupyter kernel is planned separately.

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

By default AutoVibe Gym keeps the existing OpenAI-compatible client. This covers
local vLLM, OpenAI, LiteLLM, and other proxies that expose `/v1/chat/completions`:

```bash
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=local
LLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
```

For Google AI Studio / Gemini on a laptop, switch the provider and set a token:

```bash
LLM_PROVIDER=google
GEMINI_API_KEY=your-token
LLM_MODEL=gemini-2.5-flash
```

`LLM_PROVIDER=gemini` and `GOOGLE_API_KEY` are accepted aliases. If `LLM_MODEL`
is omitted, the default model is `Qwen/Qwen2.5-Coder-7B-Instruct` for the
OpenAI-compatible provider and `gemini-2.5-flash` for Google.

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

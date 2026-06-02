# AutoML Gym — Reproducibility Guide

## Prerequisites

- Docker with the `autovibe-gym` image built (see `Dockerfile`)
- Access to an OpenAI-compatible LLM endpoint
- Python 3.10+ with `paramiko` for remote runs (optional)

### Build the Docker image

```bash
docker build -t autovibe-gym .
```

### Environment variables

Copy `.env.example` to `.env` and fill in:

```bash
LLM_BASE_URL=http://llm.letovo.site:8809/openai   # or any OpenAI-compatible endpoint
LLM_API_KEY=<your-key>
LLM_MODEL=deepseek-v4-flash                        # or groq/llama-3.3-70b-versatile etc.
MLFLOW_TRACKING_URI=file:///autovibe/mlruns        # or mlflow server URL
```

LiteLLM provider routing: if `LLM_MODEL` contains `/` (e.g. `groq/llama-3.3-70b-versatile`),
the client automatically uses the LiteLLM Python SDK instead of the OpenAI-compatible adapter.

---

## Repeating an Experiment Run

### 1. Prepare a dataset

```bash
# Inside the container (or with --dataset flag pointing to a prepared dir)
python -m scripts.prepare_datasets --dataset student_dropout
```

This reads `datasets/student_dropout/config.json`, loads raw data from
`datasets/student_dropout/raw_data/`, and writes:
- `datasets/student_dropout/prepared/train.csv`
- `datasets/student_dropout/prepared/val.csv`
- `datasets/student_dropout/prepared/test.csv`
- `datasets/student_dropout/prepared/meta.json`

Splits are **deterministic**: same seed → same rows in train/val/test every time.

### 2. Run an experiment

All four modes share the same CLI pattern:

```bash
# Single-shot baseline
python -m experiments.run_baseline \
  --dataset-dir datasets/student_dropout/prepared \
  --mode local \
  --model deepseek-v4-flash

# Repeated single-shot
python -m experiments.run_multishot \
  --dataset-dir datasets/student_dropout/prepared \
  --mode local --shots 5

# Flexible transitions (gym)
python -m experiments.run_gym \
  --dataset-dir datasets/student_dropout/prepared \
  --mode local --max-steps 15

# Fixed transitions
python -m experiments.run_fixed \
  --dataset-dir datasets/student_dropout/prepared \
  --mode local
```

`--mode local` uses 30-step budget and 60s sandbox timeout.
`--mode cloud` uses 15-step budget and 30s sandbox timeout.

To run the full four-mode product matrix across datasets and models:

```bash
python -m experiments.run_all_modes_matrix \
  --datasets datasets/student_dropout/prepared \
  --models deepseek-v4-flash gemma-4-26b \
  --mode cloud \
  --experiment-name autovibe-gym
```

### 3. Inside Docker (as used on the H200 server)

```bash
docker run --rm \
  -v /path/to/autovibe-gym:/autovibe \
  -e LLM_BASE_URL=http://... \
  -e LLM_API_KEY=... \
  -e LLM_MODEL=deepseek-v4-flash \
  -e MLFLOW_TRACKING_URI=file:///autovibe/mlruns \
  autovibe-gym \
  -m experiments.run_baseline \
  --dataset-dir /autovibe/datasets/student_dropout/prepared \
  --mode local
```

### 4. View results

```bash
# Compare all runs in a dataset
python -m experiments.compare --dataset student_dropout

# Or launch MLflow UI
mlflow ui --backend-store-uri mlruns/ --port 5000
```

---

## Adding a New Tabular Dataset

The project now has one shared ingestion system for CLI preparation and the web
dashboard. Inputs may be raw or already split, may come from local files or
HTTP(S) URLs, and may use multiple related tables.

### Supported input formats

- `.csv`
- `.csv.gz`
- `.tsv`
- `.txt` with delimiter detection or explicit `read_options.sep`
- `.parquet`
- `.xlsx`
- `.xls`
- `.json`
- `.jsonl`
- `.ndjson`
- `.zip` containing supported tabular members
- optional when deps are installed: `.feather`, `.orc`

### Minimal raw-mode config

```yaml
name: my_dataset
suite: custom

source:
  title: "Dataset title"
  url: ""
  license: ""
  citation: ""
  description: ""

dataset_notes:
  short_description: "Short human summary"
  llm_context: |
    Optional safe context for the LLM. This can include domain meaning,
    leakage warnings, or even the human-written task text.
  warnings: []
  known_pitfalls: []

ingestion:
  mode: raw
  files:
    - logical_name: table_1
      role: base
      source_type: local
      url: ""
      path: raw_data/data.csv
      format: auto
      read_options:
        sep: ","
        encoding: utf-8
      optional: false
      archive_member: ""

relations:
  base_table: table_1
  joins: []

task:
  type: classification
  target_col: label
  metric: f1_macro
  forbidden_columns: []

split:
  strategy: stratified_random
  seed: 42
  train_fraction: 0.7
  val_fraction: 0.15
  test_fraction: 0.15

preparation:
  drop_columns: []
  rename_columns: {}
  target_mapping: {}
```

### Already split mode

Use `ingestion.mode: pre_split` and declare file roles `train`, `test`, and
optionally `val`. When `val` is missing, set:

```yaml
split:
  strategy: pre_split
  seed: 42
  create_val_from_train_if_missing: true
  val_fraction_from_train: 0.15
```

### Multi-table relational mode

Declare multiple `ingestion.files` entries and connect them with
`relations.base_table` plus `relations.joins`. The loader validates join keys,
records row counts before/after joins, and warns or errors when joins multiply
rows too aggressively.

### Prepare splits

```bash
python3 scripts/prepare_datasets.py --dataset my_dataset
```

Prepared output stays compatible with existing experiment runners:

```text
datasets/my_dataset/prepared/
  train.csv
  val.csv
  test.csv
  meta.json
```

`meta.json` now includes `source`, `dataset_notes`, input file metadata, join
diagnostics, warnings, and forbidden/dropped columns in addition to the task
metadata used by the runners.

### Dashboard flow

The dashboard can now:

- create/edit dataset configs;
- upload raw files;
- download raw files from URL into `raw_data/`;
- preview individual tables or the joined modeling dataframe;
- validate config and joins before preparation;
- prepare the standard `prepared/` layout from the UI.

Human users may preview uploaded data in the dashboard. Agent-facing runtimes
still receive only `train_df`, `val_df`, `target_col`, and safe metadata; they
do not receive hidden test labels, hidden metrics, private evaluator artifacts,
or hidden checklist answers.

### Run experiments

```bash
python -m experiments.run_baseline \
  --dataset-dir datasets/my_dataset/prepared --mode local
```

No other code changes are needed. The environment reads task type, metric, and
target column from `meta.json` at runtime.

---

## Determinism Notes

- All splits use `sklearn.train_test_split` with `random_state=seed` from config.
- Temporal splits are purely order-based (no randomness).
- LLM responses are **not** deterministic — re-running the same experiment with the
  same model may produce different scores. To compare modes fairly, run each mode
  at least once on the same dataset and record all runs in MLflow.
- The private test set is evaluated exactly once per run (enforced by `GymEnv.submit()`).
  There is no way to query the test score multiple times within a single session.
- `null` test metrics mean no valid hidden-test submission was produced. They
  are represented as missing metrics plus `final_status` / `null_reason`, never
  as fake `0.000` scores.
- Profiling and diagnostics tools are optional. `profile_data` always has a
  compact pandas backend; ydata-profiling, cleanlab, and optuna live in the
  `diagnostics` extra and are used only when requested/configured.

---

## Artifact Locations

| Artifact | Path (in MLflow run) |
|----------|----------------------|
| Full cell-by-cell notebook | `cell_history.md` |
| Stage-level log (fixed mode) | `stage_log.json` |
| Attempt-level log (repeated SS) | `attempt_log.json` |
| MLflow run metadata | `mlruns/<experiment_id>/<run_id>/` |

Private gym artifacts may also include:

- `agent_trace_private.jsonl`
- `cell_executions_private.jsonl`
- `candidate_diagnostics_private.jsonl`
- `data_profile_private.json`
- `data_profile_ydata.json` / `data_profile_ydata.html`
- `cleanlab_diagnostics_private.json`

These private artifacts are not mounted into the agent-visible workspace.

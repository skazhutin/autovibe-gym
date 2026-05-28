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

### Step 1 — Add raw data

Place the raw file(s) under `datasets/<name>/raw_data/`:

```
datasets/
  my_dataset/
    raw_data/
      data.csv
```

### Step 2 — Write `config.json`

```json
{
  "name": "my_dataset",
  "suite": "example_datasets",
  "source": {"type": "local_file", "provider": "...", "license": "..."},
  "raw_data": {
    "files": ["data.csv"],
    "format": "csv",
    "read_options": {"sep": ",", "encoding": "utf-8"}
  },
  "task": {
    "type": "classification",
    "target_col": "label",
    "metric": "f1_macro"
  },
  "split": {
    "strategy": "stratified_random",
    "seed": 42,
    "train_fraction": 0.7,
    "val_fraction": 0.15,
    "test_fraction": 0.15
  },
  "preparation": {
    "drop_columns": [],
    "deduplicate": false
  },
  "role": "short_description_of_dataset_type",
  "notes": {}
}
```

**Supported split strategies:**
- `stratified_random` — class-balanced random split (classification)
- `temporal` — chronological split using a timestamp column (requires `split.timestamp` block)

**Supported preparation options:**
- `drop_columns`: list of column names to remove before splitting
- `deduplicate`: `true` removes exact duplicate rows before splitting
- `target_mapping`: dict to remap target values (e.g. `{"0": "neg", "1": "pos"}`)
- `rename_columns`: dict to rename columns
- `sampling.allowed: true` + `sampling.strategy`: enables `--max-rows` sampling

### Step 3 — Prepare splits

```bash
python -m scripts.prepare_datasets --dataset my_dataset
```

Verify the output:

```
datasets/my_dataset/prepared/
  train.csv    val.csv    test.csv    meta.json
```

### Step 4 — Run experiments

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

---

## Artifact Locations

| Artifact | Path (in MLflow run) |
|----------|----------------------|
| Full cell-by-cell notebook | `cell_history.md` |
| Stage-level log (fixed mode) | `stage_log.json` |
| Attempt-level log (repeated SS) | `attempt_log.json` |
| MLflow run metadata | `mlruns/<experiment_id>/<run_id>/` |

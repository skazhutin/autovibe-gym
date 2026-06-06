# AutoVibe Gym — Release / Demo Checklist

Use this checklist before running a demo or sharing results externally.

---

## Environment

- [ ] The needed LLM models exist in the shared model registry (`python -m experiments.models list`)
- [ ] `.env` has no LLM model settings or LLM API keys
- [ ] `MLFLOW_TRACKING_URI` is set (or defaults to `sqlite:///mlflow.db`)
- [ ] `AUTOVIBE_KERNEL_BACKEND=docker` if using the container sandbox
- [ ] Docker daemon running and `autovibe-gym-sandbox:latest` built
  ```bash
  docker build -f Dockerfile.sandbox -t autovibe-gym-sandbox:latest .
  ```

## Datasets

- [ ] At least one dataset has been prepared:
  ```bash
  python scripts/prepare_datasets.py --dataset-config datasets/example_student_dropout/config.yaml
  ```
- [ ] `datasets/<name>/prepared/{train,val,test}.csv` and `meta.json` exist
- [ ] `test.csv` is NOT accessible to the agent (verify: it is not in the episode workspace)

## Tests

- [ ] Unit + integration suite passes:
  ```bash
  python -m pytest tests/ -q
  ```
- [ ] (Optional) Docker integration test passes:
  ```bash
  python -m pytest tests/test_jupyter_kernel.py -m integration -v
  ```
- [ ] Docker sandbox smoke test passes:
  ```bash
  python scripts/verify_docker_sandbox.py
  ```

## Single run

- [ ] Baseline single-shot runs end-to-end:
  ```bash
  python -m experiments.run_baseline --dataset-dir datasets/example_student_dropout --mode local
  ```
- [ ] Gym run produces MLflow artifacts (cell_history.md, episode_summary.json):
  ```bash
  python -m experiments.run_gym --dataset-dir datasets/example_student_dropout --mode local
  ```

## Batch matrix

- [ ] Dry-run preview is correct:
  ```bash
  python -m experiments.run_matrix --model deepseek-v4-flash --dry-run
  ```
- [ ] Full matrix completes without errors:
  ```bash
  python -m experiments.run_matrix --mode local --model deepseek-v4-flash
  ```

## MLflow

- [ ] Results visible in MLflow UI:
  ```bash
  mlflow ui --host 0.0.0.0 --port 5000
  ```
- [ ] `compare.py` produces a summary table:
  ```bash
  python -m experiments.compare --output results.csv
  ```

## Privacy checks

- [ ] No `test.csv` path in any agent-facing message (grep `cell_history.md`)
- [ ] No hidden score in `feedback_trace.json` (value should be absent or `null`)
- [ ] Kernel env does not contain `OPENAI_API_KEY`, `LLM_API_KEY` etc.
  (verified by `_minimal_kernel_env()` and integration test)

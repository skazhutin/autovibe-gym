# AutoVibe Gym — Technical Backlog

Items that are known gaps but intentionally out of scope for the current phase.
Each item has a priority label and a brief rationale for deferral.

---

## 🔴 High priority

### Full OS isolation for Jupyter kernel (container backend)
**Status:** `ContainerJupyterKernelBackend` implemented (PR #11). Integration test
exists but requires Docker daemon + built sandbox image.
**What remains:** Run `pytest -m integration` on a machine with Docker to get a
non-skipped pass. Document result in EXPERIMENT_REPORT.

### Fresh notebook-era experiment matrix
**Status:** Historical experiment results are from the legacy subprocess runtime.
The notebook-era runtime (`NotebookGymEnv`) is production-ready but no full
experiment matrix has been run through it yet.
**What remains:** Run `python -m experiments.run_matrix --mode local` on the H200
server against student_dropout and room_occupancy to produce a fresh experiment
report comparable to Section 1 and 2 of EXPERIMENT_REPORT.md.

---

## 🟡 Medium priority

### Non-skipped Docker integration CI
**Status:** `.github/workflows/docker-sandbox.yml` added. Will trigger on push to
main when Dockerfile.sandbox or jupyter_kernel.py changes.
**What remains:** Verify the workflow passes on GitHub Actions (requires a
self-hosted runner or standard runner with Docker available).

### LLM-judge checklist (ADR-006 v2)
**Status:** Deferred. Current regex checklist works for experiments.
**What remains:** Design evaluation, pick judge model, add optional `--judge-model` flag.

### `run_fixed.py` notebook-era port
**Status:** `run_fixed.py` still uses the legacy `GymEnv` / `CodeExecutor` stack.
It is marked as legacy in MLflow params.
**What remains:** Port to `NotebookGymEnv` if fixed-stage experiments are needed
in the notebook era. Low priority — fixed transitions scored worst in all experiments.

---

## 🟢 Low priority

### Multi-model comparison
**Status:** Framework supports any OpenAI-compatible model. No systematic
multi-model runs have been done yet.
**What remains:** Run `run_matrix.py` with `--model` variants for Qwen-7B, Qwen-32B,
Qwen-72B-AWQ.

### Persistent kernel across episodes
**Status:** Each `NotebookGymEnv` episode creates a fresh kernel. This is correct
for isolation but means conda/pip install inside one episode doesn't carry over.
**What remains:** Assess whether this is a real limitation for any planned experiment.

### Web UI for experiment comparison
**Status:** `compare.py` produces a CSV/terminal table. MLflow UI is available
at `http://<server>:5000`.
**What remains:** Nothing blocking. MLflow UI is sufficient for current needs.

# AutoVibe Gym - Live Status

**Last updated:** 2026-05-29 (ContainerJupyterKernelBackend — Docker sandbox for Jupyter kernels)
**Phase:** PR #11 open. 133 tests passing. ContainerJupyterKernelBackend awaiting merge.

---

## Current Sprint Goal

Move iterative Gym experiments from simulated notebook execution to a real
Jupyter `.ipynb` + persistent kernel environment while preserving hidden-test
privacy and fair checklist ablations.

---

## Status by Component

### Core Gym (`gym/`)

| File | Status | Notes |
|------|--------|-------|
| `notebook.py` | Done | nbformat v4 document editing, stable cell ids, revisions, outputs, Python export |
| `jupyter_kernel.py` | Done | persistent local `ipykernel` via `jupyter_client`, bootstrap injection, rich output/error capture |
| `notebook_env.py` | Done | real notebook action loop, clean restart-and-run-all, validate, submit, artifacts |
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
| `run_baseline.py` | Done | single-shot control preserved; missing score is not logged as zero |
| `run_multishot.py` | Done | logged as `repeated_single_shot`; not the fair checklist control |
| `compare.py` | Done | handles missing metrics without zero substitution |

### Privacy and Security

| Item | Status | Notes |
|------|--------|-------|
| Hidden test files | Done | not copied into episode workspace; no `test_df` in kernel |
| Hidden score feedback | Done | submit response hides score; score only in private summary/MLflow |
| Local Jupyter sandbox | Limited | real notebook functionality, sanitized env, but not full OS isolation |
| Future container backend | Planned | `ContainerJupyterKernelBackend` placeholder exists |

### Tests

| Area | Status |
|------|--------|
| Existing legacy env/executor/agent tests | Passing |
| Jupyter kernel tests | Passing |
| Notebook editing tests | Passing |
| Clean run / validate / submit tests | Passing |
| Checklist privacy/fairness tests | Passing |
| Hidden-test privacy tests | Passing |

---

## Current Verification

Last local run:

```bash
python -m compileall gym experiments tests scripts
python -m pytest tests
python -m pytest --cov=gym --cov=experiments --cov=scripts --cov-report=term-missing --cov-report=xml --cov-fail-under=70
git diff --check
```

Result after rebase: `126 passed`, coverage `72.76%`.

---

## Blocked / Needs Decision

- Full container isolation for the Jupyter kernel is intentionally out of scope
  for this PR. Hidden-test physical isolation is implemented now; network and
  arbitrary filesystem hardening for local Jupyter remains future work.
- Existing `GymEnv` remains for compatibility, but new iterative experiments
  should use `NotebookGymEnv`.

---

## Next Actions

1. [x] Завершить эксперименты room_occupancy — все 4 режима запущены и завершены
2. [x] Обновить EXPERIMENT_REPORT.md с результатами
3. [x] Смержить эксперименты в main — commit 638fcd6
4. [x] Добавить real Jupyter notebook environment (PR #9) — merged to main
5. [ ] Смержить ContainerJupyterKernelBackend (PR #11) — ожидает ревью

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-29 | Implemented ContainerJupyterKernelBackend: Docker sandbox with --internal network, --read-only rootfs, --cap-drop ALL; PR #11 open |
| 2026-05-29 | Rebasing Jupyter branch on updated `origin/main` and preserving LiteLLM/Groq provider support |
| 2026-05-29 | Added real Jupyter `.ipynb` + persistent kernel backend with notebook actions |
| 2026-05-29 | Added clean `restart_and_run_all`, host-controlled `validate`, candidate registry, and submit gate |
| 2026-05-29 | Split feedback channels and replaced dataset-specific checklist hints with generic selective hints |
| 2026-05-29 | Hid hidden test metric from agent-facing messages, feedback traces, and notebook outputs |
| 2026-05-29 | Added `iterative_no_checklist` as fair Jupyter control and renamed multishot logging to `repeated_single_shot` |
| 2026-05-29 | Added Jupyter dependencies and tests for kernel, notebook edits, clean replay, validation, privacy, and fairness |

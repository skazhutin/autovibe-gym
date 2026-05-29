# AutoML Gym — Interaction Protocol Specification

## Overview

The AutoML Gym evaluates LLM agents on supervised tabular ML tasks through a controlled
interaction protocol. The environment enforces privacy boundaries, tracks agent trajectories,
and supports four distinct interaction modes to separate the contribution of different
feedback signals.

---

## Data Splits and Privacy Boundary

```
datasets/<name>/prepared/
  train.csv   — visible to agent code
  val.csv     — visible to agent code
  test.csv    — NEVER visible to agent code; used only by env.submit()
  meta.json   — task metadata (target_col, metric, split_strategy, …)
```

**Privacy guarantee:** `CodeExecutor` runs all agent code in a subprocess with
`cwd=tmpdir` (a temporary directory that contains no data files). The agent cannot
access `test.csv` via relative paths or by traversing the filesystem from the working
directory.

**Final evaluation:** `env.submit(model)` calls `model.predict(X_test)` exactly once.
The result is irreversible. The private test score must not be used to select the model —
only validation scores may inform that decision.

---

## Action Protocol

Every agent action is a JSON object. Two action types are defined:

```json
{"type": "code", "code": "print(train_df.shape)"}
{"type": "submit", "model_var": "model"}
```

**Code action:** The code string is executed in a persistent namespace containing
`train_df`, `val_df`, `target_col`, `pd`, `np`. Outputs (stdout/stderr) and updated
namespace are returned to the agent.

**Submit action:** Evaluates the named workspace variable on the private test set.
Closes the environment. Only one submit is allowed per session.

---

## Candidate Lifecycle

```
Train → Validate → Choose → Replay → Final
```

| Step | Who acts | Description |
|------|----------|-------------|
| **Train** | Agent | Builds a model via code actions; assigns to workspace variable |
| **Validate** | Agent | Evaluates `model.predict(val_df)` inside code; records val score |
| **Choose** | Agent or env | Agent issues `{"type": "submit", "model_var": "…"}` explicitly; or env auto-submits the first workspace variable with `predict()` on budget exhaustion |
| **Replay** | Env | `model.predict(X_test)` called on raw held-out rows — no manual preprocessing; model must be self-contained |
| **Final** | Env | One-time private test evaluation; score logged to MLflow |

**Fallback (legacy `GymEnv` path):** If the agent does not submit before budget exhaustion,
the environment scans the workspace for any object with a `predict()` method (in order:
`best_model`, `model`, then any other name) and auto-submits it.

**Fallback (`NotebookGymEnv` path — stricter contract):** The notebook environment requires
an explicit `validate` action before `submit`. If the budget is exhausted without a
successful `validate`, the forced submit attempt returns a `NEEDS_VALIDATION` blocker and
`test_metric` remains `null`. This is intentional: without a confirmed `restart_and_run_all`
clean replay, the environment cannot guarantee raw-input reproducibility on the hidden test
split. The agent is expected to complete at least one `validate` cycle within its step budget.

---

## Interaction Modes

### Mode 1 — Single-shot

| Property | Value |
|----------|-------|
| Runner | `experiments/run_baseline.py` |
| LLM calls | 1 |
| Feedback to agent | None |
| Checklist hints | No |
| Traceback shared | No |
| Namespace shared | N/A |

The agent writes a complete solution in one response. No iteration, no feedback.
Tests the raw generation strength of the model.

---

### Mode 2 — Repeated Single-shot

| Property | Value |
|----------|-------|
| Runner | `experiments/run_multishot.py` |
| LLM calls | N (default 10 local / 5 cloud) |
| Feedback between attempts | Only: best validation metric so far (scalar) |
| Checklist hints | No |
| Traceback shared | No |
| Namespace shared | No — each attempt starts from a fresh namespace |

Each attempt is fully independent. The only inter-attempt signal is:
`"Previous best validation score: 0.74"`. No stdout, no stderr, no stage hints.

Tests whether knowing a scalar "can you beat X?" signal helps generation.

---

### Mode 3 — Fixed Transitions

| Property | Value |
|----------|-------|
| Runner | `experiments/run_fixed.py` |
| LLM calls | Up to 30 local / 15 cloud |
| Stage order | Fixed: EDA → Preprocessing → Feature Engineering → Model Selection → HPT |
| Feedback per step | stdout + stderr + checklist hints |
| Checklist hints | Yes |
| Stage skipping | Not allowed |

The pipeline is divided into 5 mandatory stages with per-stage budgets.
After each stage's budget is exhausted the runner advances automatically.
The agent receives full checklist feedback within each stage but cannot
choose the order or reopen a prior stage.

**Local stage budgets:** EDA=3, Preprocessing=4, Feature Engineering=4,
Model Selection=6, HPT=5 (total=22 + 5 buffer).

---

### Mode 4 — Flexible Transitions

| Property | Value |
|----------|-------|
| Runner | `experiments/run_gym.py` |
| LLM calls | Up to 30 local / 15 cloud |
| Stage order | Agent decides |
| Feedback per step | stdout + stderr + checklist hints + notebook context |
| Checklist hints | Yes |
| Autonomous refinement | Yes |

The agent freely decides what to do next at each step. Full checklist feedback
after every code action. No stage structure imposed.

---

## Feedback Taxonomy

| Signal | Single-shot | Repeated SS | Fixed | Flexible |
|--------|:-----------:|:-----------:|:-----:|:--------:|
| Task description | ✅ | ✅ | ✅ | ✅ |
| Previous best val score | ❌ | ✅ | ❌ | ❌ |
| Code execution stdout | ❌ | ❌ | ✅ | ✅ |
| Code execution stderr | ❌ | ❌ | ✅ | ✅ |
| Checklist hints | ❌ | ❌ | ✅ | ✅ |
| `[MODEL CHECK]` pre-flight hint | ❌ | ❌ | ✅ | ✅ |
| Stage goal announcement | ❌ | ❌ | ✅ | ❌ |
| Notebook context (prior cells) | ❌ | ❌ | ✅ | ✅ |
| Private test score | ❌ | ❌ | ❌ | ❌ |

---

## Checklist

The environment implicitly tracks 8 DS pipeline stages and returns hints (not
instructions) when a stage has not been addressed. Hints are returned only in
interactive modes (Fixed and Flexible).

| Stage | Hint text |
|-------|-----------|
| EDA | "Have you explored the dataset before modelling?" |
| Missing values | "Some columns may have gaps — worth addressing." |
| Duplicates | "Duplicate rows can silently bias training." |
| Target leak | "Make sure no feature is a direct proxy of the target." |
| Train/val split | "It helps to evaluate on a held-out validation set." |
| Feature engineering | "Raw features aren't always optimal — tried transforms?" |
| Model selection | "Have you compared more than one model type?" |
| Hyperparameter tuning | "Default hyperparameters are rarely optimal." |

Hint philosophy: hints point at *what* to consider, never *how* to do it.
`"Consider checking for duplicate rows"` not `"Call drop_duplicates()"`.

---

## Trajectory Audit

Every run produces the following MLflow artifacts:

| Artifact | Content |
|----------|---------|
| `cell_history.md` | Markdown notebook: each code cell with stdout/stderr/hints |
| `stage_log.json` | (Fixed mode only) per-stage step count, error count, checklist coverage |
| `attempt_log.json` | (Repeated SS only) per-attempt val metric and error flag |
| MLflow params | experiment_type, mode, model, dataset, max_steps, budgets |
| MLflow metrics | test_metric, checklist_coverage, steps_used, error_count, tokens, elapsed |

---

## Environment Guarantees

1. **No test leakage:** subprocess runs with `cwd=tmpdir`; test split never injected into namespace.
2. **Single final evaluation:** `env.submit()` raises `RuntimeError` on second call.
3. **Reproducible splits:** fixed seed, deterministic stratified split via `sklearn.train_test_split`.
4. **Sandbox timeout:** each code step has a configurable timeout (default 60s local, 30s cloud).
5. **Label-type robustness:** `env.submit()` applies a three-stage coercion chain to
   handle prediction/label dtype mismatches (e.g. agent returns int predictions for
   string labels): (a) cast to `y_test.dtype`, (b) map int indices to sorted class
   labels, (c) raise descriptive error. This makes the final evaluation robust to
   common label-encoding patterns without silently altering predictions.
6. **Pre-flight model validation (ADR-010):** After every code step in interactive modes,
   the environment calls `model.predict(X_val[:32])` on raw validation rows for each
   model variable in the workspace (`model`, `best_model`). Failures return a
   `[MODEL CHECK]` hint identifying the variable and error before any submit is attempted.
   `submit_by_name()` repeats the same check at submit time: if pre-flight fails, the
   submit is **blocked** (`submitted=False` preserved) and the agent can correct the
   pipeline within the remaining budget. This guarantees the one-time hidden test
   evaluation is never consumed by a structurally broken model.

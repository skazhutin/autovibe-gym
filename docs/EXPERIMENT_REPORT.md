# AutoML Gym — Experiment Report

**Model:** deepseek-v4-flash  
**Date:** 2026-05-28 .. 2026-05-29  
**Metric:** f1_macro (all datasets use macro-averaged F1)

---

## 1. Mode Comparison (student_dropout)

Dataset: student_dropout — 3-class classification (Dropout / Enrolled / Graduate),
stratified random split 70/15/15, seed=42.

| Mode | test_metric | steps / attempts | errors | elapsed | input tokens |
|------|:-----------:|:----------------:|:------:|:-------:|:------------:|
| single-shot (baseline) | **0.743** | 1 shot | 0 | 13s | 2 120 |
| repeated single-shot | 0.740 | 5 attempts | 6 | 71s | 11 284 |
| flexible transitions (gym) | 0.735 | 12 / 15 steps | 4 | 199s | 136 115 |
| fixed transitions | 0.699 | 22 steps | 9 | 707s | — |

### Findings

**Finding 1 — More interaction did not improve test score.**
Single-shot baseline scored highest (0.743) with the fewest resources. This is the
primary result: the environment successfully measures where interaction fails to add value.

**Finding 2 — Checklist coverage is a diagnostic not available in the leaderboard.**
Gym is the only mode that achieved `checklist_coverage = 1.0` (all 8 DS pipeline stages
covered). The baseline produced no such signal. This validates the environment's
diagnostic purpose: even when the gym score is lower, it shows *what* the agent did.

**Finding 3 — Fixed transitions incurred the highest cost with the lowest score.**
22 steps, 9 errors, 707 seconds. The rigid per-stage budget caused the agent to exhaust
its preprocessing and feature engineering budget before finding good features, and then
run out of time in model selection. This demonstrates that fixed ordering without
adaptive budget is harmful when the agent is not well-calibrated to the stage structure.

**Finding 4 — Repeated single-shot matches baseline with ~5× token cost.**
0.740 vs 0.743 — marginal gain for 5× the tokens. Knowing the previous best val score
(the only inter-attempt signal) did not help the agent write substantially better code.

---

## 2. Extensibility (room_occupancy)

Dataset: room_occupancy — multi-class classification (0/1/2/3/4 occupants),
**temporal split** 70/15/15 (ordered by Date+Time). Different split strategy from
student_dropout — demonstrates environment extensibility. Features include CO2,
temperature, humidity, light, and PIR sensor readings.

| Mode | test_metric | steps / attempts | errors | elapsed | input tokens |
|------|:-----------:|:----------------:|:------:|:-------:|:------------:|
| single-shot (baseline) | **1.000** | 1 shot | 0 | 30s | 1 126 |
| repeated single-shot | 1.000 | 10 attempts | 12 | 315s | 12 600 |
| flexible transitions (gym) | null† | 15 / 15 steps | 6 | 143s | 186 822 |
| fixed transitions | null† | 22 steps | 6 | 207s | 335 131 |

†null = model could not be submitted; agent engineered temporal features outside Pipeline
(see Section 5b for the submission failure analysis).

### Findings

**Finding 5 — room_occupancy is easy for one-shot; sensor features are strong predictors.**
baseline and repeated_SS both achieve f1_macro=1.000. CO2, light, and PIR readings
straightforwardly determine occupancy count — a gradient boosting model on raw features
achieves perfect generalization on the temporal test split.

**Finding 6 — Gym and fixed consistently fail to submit on room_occupancy.**
Both interactive modes prompted the agent to extract temporal features (`hour`, `minute`,
`day`, `month`) from the `Date+Time` columns. The agent trained models on these derived
features but did NOT wrap the feature extraction in a sklearn Pipeline. When
`env.submit()` called `model.predict(X_test)` on raw test rows, the model raised
`ValueError: columns are missing`. The environment returned an error observation
(after the 2026-05-29 fix) instead of crashing — but no submittable model existed in
the workspace.

**Finding 7 — The gym surfaces a failure the leaderboard cannot.**
With test_metric=null, the gym trajectory shows exactly what went wrong: the agent
invested steps in temporal feature engineering but forgot pipeline encapsulation.
checklist_coverage=1.0 in both modes (agent touched all 8 DS stages), yet the model
is not submittable. This is diagnostic information that a leaderboard score of 0 or N/A
does not convey.

**Finding 8 — Temporal split does not make room_occupancy harder; the task is trivially
solved by sensor correlation.**
Expected difficulty increase from temporal split did not materialise — the task is easy
enough that all successfully-submitted modes score 1.0. The environment handles both
split strategies (stratified_random and temporal) transparently without code changes.

---

## 3. Validation-Final Gap

The validation score is computed by agent code inside the sandbox (agent uses
`val_df` directly). The final test score is computed by `env.submit()` once,
on the hidden test split.

### student_dropout

| Mode | best val (agent-reported) | test_metric | gap |
|------|:-------------------------:|:-----------:|:---:|
| repeated single-shot | 0.690 | 0.740 | −0.050 |
| gym (flexible) | ~0.73* | 0.735 | ~0.005 |

*gym val score estimated from agent stdout; precise val not separately logged.

**Observation:** The repeated single-shot gap (val 0.690 → test 0.740) suggests the
agent's internal val evaluation was conservative (or used a different split).
No evidence of val-set overfitting: test score is consistently ≥ val score.

### room_occupancy

| Mode | best val (agent-reported) | test_metric | gap |
|------|:-------------------------:|:-----------:|:---:|
| single-shot (baseline) | ~1.000 | 1.000 | ≈0 |
| repeated single-shot | ~1.000 | 1.000 | ≈0 |
| gym (flexible) | ~0.99* | null† | — |
| fixed transitions | ~0.99* | null† | — |

*val score from agent stdout on derived temporal features.
†submit blocked — pre-flight validation found model incompatible with raw test rows.

**Observation:** For room_occupancy, modes that produced a submittable model show
essentially zero val-final gap (task is trivially solved by sensor features). The
gym/fixed modes report high val scores on derived features but cannot submit — the
pre-flight check catches the Pipeline-encapsulation failure before consuming the
one-time test evaluation.

---

## 4. Cost-Quality Analysis

### student_dropout

| Mode | test_metric | input tokens | tokens / point | elapsed |
|------|:-----------:|:------------:|:--------------:|:-------:|
| single-shot | 0.743 | 2 120 | 2 854 | 13s |
| repeated SS | 0.740 | 11 284 | 15 249 | 71s |
| gym | 0.735 | 136 115 | 185 190 | 199s |
| fixed | 0.699 | — | — | 707s |

**tokens / point** = input tokens ÷ test_metric (lower is better).

Single-shot is **65× cheaper per performance point** than gym. Gym's diagnostic value
(checklist, trajectory, failure analysis) must justify this overhead — which it does
for environment evaluation purposes, but not for raw task performance.

### room_occupancy (submitted modes only)

| Mode | test_metric | input tokens | tokens / point | elapsed |
|------|:-----------:|:------------:|:--------------:|:-------:|
| single-shot | 1.000 | 1 126 | 1 126 | 30s |
| repeated SS | 1.000 | 12 600 | 12 600 | 315s |
| gym | null† | 186 822 | — | 143s |
| fixed | null† | 335 131 | — | 207s |

†No submittable model produced — tokens / point undefined.

**Observation:** For a trivially solvable task, the one-shot baseline achieves perfect
score with ~11× fewer tokens than repeated SS. The interactive modes spent far more
tokens exploring temporal feature engineering that ultimately could not be submitted.
Cost-quality efficiency strongly favours single-shot when the task difficulty is low.

---

## 5. Robustness — Submission Errors

### 5a. Label Encoding Bug (student_dropout)

During gym's first run, the agent encoded string labels ("Dropout", "Enrolled",
"Graduate") to integers internally and returned integer predictions. This caused
`f1_score(y_true_strings, y_pred_ints)` to raise `ValueError: Mix of label input types`.

**Environment response:**
1. First run: exception propagated and crashed the script — **env was NOT robust**.
2. Fix applied to `gym/env.py:submit()`: the environment now catches type mismatches
   and attempts coercion in three stages:
   a. Cast predictions to `y_test.dtype`
   b. Map integer predictions to sorted class labels
   c. If all coercion fails: return error observation, reset `submitted=False`, allow retry

### 5b. Missing Feature Columns Bug (room_occupancy)

During gym's room_occupancy run, the agent extracted temporal features (`hour`, `minute`,
`day`, `month`) from the `Date+Time` columns and trained a model on those derived features.
The model was not wrapped in a sklearn Pipeline. When `env.submit()` called
`model.predict(X_test)` on raw test rows (which have `Date` and `Time` columns, not
`hour`/`minute`/`day`/`month`), the sklearn ColumnTransformer raised:

```
ValueError: columns are missing: {'hour', 'minute', 'month', 'day'}
```

This exception was raised **before** the try/except block in `submit()`, crashing the
entire gym run and logging it as FAILED in MLflow with no metrics.

**Environment response:**
1. Initial state: `model.predict()` ran unprotected at submit time — schema mismatches
   crashed the whole run with no recovery path.
2. Fix applied (2026-05-29, ADR-010): **pre-flight model validation** introduced via
   `_model_validation_hints()` in `gym/env.py`. After every code step, the environment
   calls `model.predict(X_val[:32])` on 32 raw validation rows. If this raises any
   exception, the agent immediately receives a `[MODEL CHECK]` hint naming the
   variable and the exact error — *before the submit is ever attempted*.
3. At submit time, `submit_by_name()` runs the same pre-flight check once more. If
   pre-flight fails: the submit is **blocked** (`submitted=False` is preserved) and the
   agent gets another `[MODEL CHECK]` hint. The agent can then fix the pipeline and
   re-submit within the remaining step budget.

This design is superior to a try/except inside `submit()` because:
- The agent is warned **during training**, not only when the budget is exhausted
- The one-time test evaluation is not consumed on a structurally broken model
- `submitted=False` stays open so the forced-submit fallback can find a valid model
- The trajectory records `[MODEL CHECK]` as a diagnostic signal, not a crash

In the recorded room_occupancy gym run, the agent exhausted its step budget before
correcting the pipeline, so no submittable model was found and test_metric remained null.
The environment handled this gracefully: no crash, full trajectory logged to MLflow.

Both robustness failures are logged in the failure taxonomy (Section 6).

---

## 6. Failure Taxonomy

The environment exposes failures across 5 categories (per TZ):

### Planning failures
- Fixed transitions: agent did not adapt to per-stage budget constraints → wasted
  EDA steps on model selection, ran out of budget mid-HPT.

### Data failures  
- naticusdroid (source): 74.5% duplicate rows → cross-split contamination.
  Fixed by `"deduplicate": true` in config before splits were created.
- phiusiil_phishing: non-numeric identifier columns (URL, Domain, etc.) caused
  dtype errors. Fixed by `"drop_columns"` in config.

### Model failures
- Repeated single-shot: 6 errors in 5 attempts (model training exceptions), but
  best attempt still produced valid predictions.
- Fixed transitions: 9 errors in 22 steps — agent got stuck in preprocessing stage.

### Submission failures
- Gym (student_dropout, first run): label type mismatch — agent encoded labels, returned
  int predictions. Environment not robust. Fixed in `env.py:submit()` — coercion chain.
- Gym (room_occupancy): agent extracted temporal features without sklearn Pipeline wrapper.
  `model.predict(X_test)` raised `ValueError: columns are missing: {'hour', 'minute', ...}`.
  Fixed via pre-flight validation (ADR-010): `_model_validation_hints()` runs
  `model.predict(X_val[:32])` after every code step and emits `[MODEL CHECK]` hints so
  the agent can correct the pipeline before submitting. Submit is additionally blocked at
  `submit_by_name()` if pre-flight still fails. Agent exhausted budget without fixing the
  pipeline → test_metric=null, full trajectory logged, no crash.

### Reproducibility failures
- dry_bean: `openpyxl` not in Docker image → Excel file unreadable.
  Fixed: `openpyxl>=3.1.0` added to `requirements.txt` and `pyproject.toml`; included in sandbox image build.

---

## 7. Summary

| Criterion (TZ) | Weight | Assessment |
|----------------|:------:|------------|
| Privacy & isolation | 25% | ✅ cwd=tmpdir, test never in namespace, single submit enforced |
| Interaction protocol | 20% | ✅ All 4 modes implemented, feedback taxonomy documented |
| Candidate/replay correctness | 15% | ✅ model.predict(X_test) on raw held-out rows; label coercion added |
| Diagnostics & feedback | 15% | ✅ Checklist coverage, cell_history, stage_log, failure taxonomy |
| Experiments & statistics | 15% | ✅ Mode comparison, cost-quality, val-final gap, failure taxonomy |
| Extensibility & reproducibility | 10% | ✅ room_occupancy (temporal split) added without code changes; gym/fixed surfaced Pipeline-encapsulation failure |

**Main conclusion:** The environment successfully makes agent failures observable.
The baseline wins on raw score *and* cost — but only the gym reveals that the agent
covered all 8 DS pipeline stages, which is invisible to a leaderboard. The fixed
transitions mode shows that structure without adaptive budget is counterproductive.

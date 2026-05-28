# AutoVibe Gym

AutoVibe Gym is an iterative AutoML environment where an LLM writes ML code,
receives structured feedback, improves the solution, and submits one final model
against a hidden test split.

## Core Loop

```text
GymAgent
  -> LLMClient
  -> JSON Action
  -> GymEnv.step()
  -> CodeExecutor + Checklist + CellHistory
  -> Observation feedback
  -> next JSON Action
```

Actions:

```json
{"type": "code", "code": "print(train_df.shape)"}
```

```json
{"type": "submit", "model_var": "best_model"}
```

The agent workspace contains `train_df`, `val_df`, `target_col`, `pd`, and `np`.
`test_df` is never exposed to code actions.

Each code action is also stored as a notebook-like cell with code, stdout,
stderr, checklist hints, and coverage. Workspace variables persist across cells,
so the agent can build on previous work instead of rewriting a full script every
turn.

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

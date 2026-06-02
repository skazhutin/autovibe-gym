# Dataset System

AutoVibe Gym datasets have two layers:

- `raw/`: source files uploaded by a user or extracted from an archive.
- `prepared/`: runner-compatible CSV splits consumed by `experiments.run_*`.

The runner contract is intentionally stable:

```text
datasets/<dataset_id>/
  prepared/
    train.csv
    val.csv
    test.csv
    meta.json
```

New dashboard-created datasets also include:

```text
datasets/<dataset_id>/
  raw/
    uploaded/
    extracted/
  dataset_config.json
```

## Supported Upload Formats

The dashboard stages and previews tabular files in these formats:

- CSV, TSV, and tabular TXT
- JSON and JSONL
- XLSX and XLS
- Parquet and Feather when the optional pandas dependencies are installed
- ZIP, TAR, TAR.GZ, TGZ, and single-file GZ archives

Archive extraction rejects absolute paths, `..` path traversal, unsafe links, too
many files, and excessive extracted size.

## Split Creation

The creation wizard supports two preparation modes.

`raw_split` reads one raw table, validates that the target column exists, and
creates train/val/test CSVs using the requested ratios, seed, shuffle flag, and
stratification mode.

`prepared_files` maps uploaded files to train/val/test. Train is required. Val
is optional and can be created from train with a validation ratio. Test is
optional, but a dataset without test is marked partial because final benchmark
scoring is unavailable.

All prepared outputs are written as CSV even when the input table is XLSX, JSON,
Parquet, or another supported table format.

## Metadata Files

`prepared/meta.json` is the compatibility file used by existing runners. It
contains the dataset name, target column, metric, task type, seed, and short
notes.

`dataset_config.json` is richer dashboard metadata. It stores task settings,
split provenance, raw file inventory, source metadata, tags, warnings, and
agent-facing notes.

Existing old-format datasets that only have `prepared/meta.json` are still
listed, editable, and usable in New Run. The dashboard synthesizes a config view
for them without changing their folder until a config edit is saved.

## Agent Notes And Leakage

Agent notes may be included in LLM-agent prompts. Keep them generic and
non-leaky:

- Do not include hidden test answers, labels, row identifiers, scores, or
  evaluator-only diagnostics.
- Do not describe transformations derived from test labels.
- Prefer task framing, column meaning, and public source context.

Internal config and agent-visible notes are stored separately in
`dataset_config.json`.

## Creation Recipes

One raw file:

1. Open Dataset Center and click Add dataset.
2. Fill basic info, target, metric, and seed.
3. Upload the source table or add it by URL.
4. Choose `Split one raw table`, select the file, set ratios, and create.

Prepared train/val/test:

1. Upload the prepared table files.
2. Choose `Map prepared files`.
3. Map train, val, and test.
4. Review and create.

Train/test without val:

1. Map train and test.
2. Leave val empty and enable `Create validation split from train`.
3. Set the validation ratio and create.

Archive:

1. Upload a ZIP/TAR/TGZ/GZ archive.
2. Click Extract.
3. Select readable extracted tables for raw split or prepared mapping.

URL:

1. Paste a direct HTTP(S) file URL in the Raw files step.
2. Download, preview, select, and continue with either split mode.

## Manual QA Checklist

- Dataset Center search filters by name, id, target, metric, source, tags, and
  description.
- Filters narrow by task, status, metric, split presence, source, and warnings.
- Sorting works for name, created/updated time, rows, and features.
- A raw CSV can be uploaded and split into train/val/test.
- Prepared train/test with missing val can create val from train.
- ZIP extraction rejects traversal paths and exposes safe extracted tables.
- Preview and column stats work for train, val, and test.
- Config, sources, and agent notes save and survive refresh.
- Existing prepared datasets still appear and can be selected in New Run.

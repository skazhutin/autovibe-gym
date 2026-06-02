# Example Datasets

This directory intentionally contains only curated example datasets for local
smoke tests, CI, and demos of the AutoVibe Gym dataset pipeline.

Only these five dataset folders are allowed to be committed:

- `example_dry_bean`
- `example_student_dropout`
- `example_room_occupancy`
- `example_naticusdroid`
- `example_phiusiil_phishing`

Each example dataset follows:

```text
example_<dataset_name>/
  config.yaml
  raw_data/
    .gitkeep
    <source archive>.zip
  prepared/        # generated locally, ignored by git
```

The `config.yaml` files and the raw zip archives for these five examples are
committed so a fresh clone can run the dataset pipeline without manual uploads.
Prepared outputs such as `train.csv`, `val.csv`, `test.csv`, and `meta.json` are
generated locally and ignored.

No other dataset folders should be committed under `datasets/`: not configs, not
raw files, not prepared splits. The repository `.gitignore` enforces this by
allowing only the five `example_*` directories above.

Commands:

```bash
python scripts/prepare_datasets.py --list
python scripts/prepare_datasets.py --dataset example_dry_bean
```

These examples are for pipeline validation, not canonical benchmark claims.

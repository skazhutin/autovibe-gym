# Datasets

Each dataset follows:

```
<dataset_name>/
  config.json
  raw_data/
  prepared/
```

- `raw_data/`: source files provided by project team.
- `prepared/`: generated `train.csv`, `val.csv`, `test.csv`, `meta.json`.
- `config.json`: declarative preparation settings.

Commands:

```bash
python scripts/prepare_datasets.py --list
python scripts/prepare_datasets.py --dataset dry_bean
python scripts/prepare_datasets.py --suite example_datasets
```

Example datasets:
- student_dropout — mixed_features_imbalanced_multiclass
- room_occupancy — temporal_sensor_leakage_discipline
- naticusdroid — high_dimensional_binary_security_domain
- phiusiil_phishing — large_mixed_features_suspicious_columns
- dry_bean — fast_numeric_multiclass_smoke_test

Preparation does not perform ML preprocessing for the agent.

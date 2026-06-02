import argparse
import json
from pathlib import Path

from gym.dataset_ingestion import (
    DatasetConfig,
    FRACTION_TOLERANCE as _FRACTION_TOLERANCE,
    apply_declared_preparation,
    config_path as _config_path,
    discover_dataset_dirs,
    load_dataset_config,
    load_raw_dataframe,
    prepare_dataset,
    raw_inputs_available,
    split_temporal,
)


DATASETS_ROOT = Path("datasets")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--suite")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    discovered = discover_dataset_dirs(DATASETS_ROOT)

    if args.list:
        print("name | suite | task_type | metric | split_strategy | raw_data_status | prepared_status | role")
        for name, ds_dir in discovered.items():
            if _config_path(ds_dir) is not None:
                cfg = load_dataset_config(ds_dir)
                raw_ok = raw_inputs_available(ds_dir, cfg)
                prepared_ok = (ds_dir / "prepared" / "meta.json").exists() or (ds_dir / "meta.json").exists()
                print(
                    f"{name} | {cfg.suite} | {cfg.task.get('type')} | {cfg.task.get('metric')} | "
                    f"{cfg.split.get('strategy')} | {'ok' if raw_ok else 'missing'} | "
                    f"{'ok' if prepared_ok else 'missing'} | {cfg.role}"
                )
            else:
                meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
                print(
                    f"{name} | legacy | {meta.get('task_type','classification')} | "
                    f"{meta.get('metric')} | {meta.get('split_strategy','fixed')} | n/a | ok | {meta.get('role')}"
                )
        return

    if args.dataset:
        to_process = [DATASETS_ROOT / args.dataset]
    elif args.suite:
        to_process = [
            ds_dir
            for _, ds_dir in discovered.items()
            if _config_path(ds_dir) is not None and load_dataset_config(ds_dir).suite == args.suite
        ]
    else:
        to_process = [ds_dir for _, ds_dir in discovered.items() if _config_path(ds_dir) is not None]

    summary = []
    for ds_dir in to_process:
        try:
            result = prepare_dataset(ds_dir, max_rows=args.max_rows)
        except Exception as exc:
            result = {
                "dataset": ds_dir.name,
                "status": "error",
                "reason": str(exc),
                "prepared_dir": str(ds_dir / "prepared"),
            }
        summary.append(result)

    print("dataset | status | reason | prepared_dir")
    for row in summary:
        print(
            f"{row['dataset']} | {row['status']} | {row.get('reason','')} | {row.get('prepared_dir','')}"
        )


if __name__ == "__main__":
    main()

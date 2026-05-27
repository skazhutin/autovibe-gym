"""
Minimal experiment runner: loads a dataset, creates the env, runs the agent, prints summary.

Usage:
    python -m experiments.run_gym --dataset datasets/wine_quality.csv --target quality
"""
import argparse
import json

import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split

from gym import GymEnv, GymAgent


def load_splits(path: str, target: str):
    df = pd.read_csv(path)
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def get_metric(target_series: pd.Series):
    n_unique = target_series.nunique()
    if n_unique <= 10:
        return (lambda y, p: f1_score(y, p, average="weighted")), "f1_weighted"
    return (lambda y, p: -mean_squared_error(y, p) ** 0.5), "neg_rmse"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    train, val, test = load_splits(args.dataset, args.target)
    metric_fn, metric_name = get_metric(train[args.target])

    env = GymEnv(
        train=train,
        val=val,
        test=test,
        target_col=args.target,
        metric_fn=metric_fn,
        metric_name=metric_name,
        max_steps=args.max_steps,
    )

    # Inject dataframes into executor namespace at start
    env.state.namespace = {
        "train_df": train.copy(),
        "val_df": val.copy(),
        "target_col": args.target,
        "pd": pd,
    }

    agent = GymAgent(env=env, model=args.model)
    summary = agent.run()

    print("\n=== Run Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

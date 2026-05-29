"""
Fixed transitions experiment runner.

The pipeline is divided into mandatory stages executed in order.
The agent gets stage-specific goals and full checklist feedback,
but cannot skip or reorder stages.

Stages: EDA → Preprocessing → Feature Engineering → Model Selection →
        Hyperparameter Tuning → Submit

Usage:
    python -m experiments.run_fixed --dataset-dir datasets/student_dropout/prepared --mode local
    python -m experiments.run_fixed --dataset-dir datasets/student_dropout/prepared --mode cloud --model deepseek-v4-flash
"""
import argparse
import json
import os
import time

from gym import GymAgent, GymEnv
from gym.agent import SYSTEM_PROMPT
from gym.datasets import DatasetSplits, load_dataset_splits, metric_from_name, resolve_metric
from gym.llm import LiteLLMClient, OpenAICompatibleLLMClient
from gym.protocol import ACTION_JSON_SCHEMA, Action, ActionParseError

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES_LOCAL = [
    {
        "name": "eda",
        "label": "Stage 1/5 — EDA",
        "goal": (
            "Explore the dataset. Check shapes, dtypes, missing values, "
            "class distribution, and any obvious anomalies."
        ),
        "budget": 3,
    },
    {
        "name": "preprocessing",
        "label": "Stage 2/5 — Preprocessing",
        "goal": (
            "Handle missing values and encode categorical columns so they are "
            "numeric. Check for and remove duplicate rows if present. "
            "Build reusable transformers that can be applied at predict time."
        ),
        "budget": 4,
    },
    {
        "name": "feature_engineering",
        "label": "Stage 3/5 — Feature Engineering",
        "goal": (
            "Create or transform features. Consider scaling, log transforms, "
            "interaction terms, or domain-specific encodings. "
            "Evaluate impact on validation score."
        ),
        "budget": 4,
    },
    {
        "name": "model_selection",
        "label": "Stage 4/5 — Model Selection",
        "goal": (
            "Train at least two different model types on the training set. "
            "Evaluate each on the validation set and record the scores. "
            "Identify the best-performing model type."
        ),
        "budget": 6,
    },
    {
        "name": "hyperparameter_tuning",
        "label": "Stage 5/5 — Hyperparameter Tuning",
        "goal": (
            "Tune the best model from the previous stage. "
            "Try at least a small parameter search. "
            "Assign the final, best model to a variable called `model`."
        ),
        "budget": 5,
    },
]

# Cloud: same stages but tighter budgets
STAGES_CLOUD = [
    {**s, "budget": max(1, s["budget"] - 1)}
    for s in STAGES_LOCAL
]

MODE_DEFAULTS = {
    "local": {"max_tokens": 8192, "sandbox_timeout": 60, "stages": STAGES_LOCAL},
    "cloud": {"max_tokens": 4096, "sandbox_timeout": 30, "stages": STAGES_CLOUD},
}


# ---------------------------------------------------------------------------
# Fixed transitions agent
# ---------------------------------------------------------------------------

def _default_llm_client(model: str):
    if "/" in model:
        return LiteLLMClient()
    return OpenAICompatibleLLMClient()


class FixedTransitionsAgent:
    """
    Wraps GymEnv and drives the agent through a fixed sequence of stages.

    At each stage transition the agent receives a new user message announcing
    the stage goal. Within a stage the feedback loop is identical to the
    flexible gym (checklist hints, stdout/stderr, budget).
    """

    def __init__(
        self,
        env: GymEnv,
        stages: list[dict],
        model: str,
        max_tokens: int = 8192,
    ):
        self.env = env
        self.stages = stages
        self.model = model
        self.max_tokens = max_tokens
        self.client = _default_llm_client(model)
        self.messages: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.stage_log: list[dict] = []
        self._start_time = time.time()

    def run(self) -> dict:
        context = self.env.reset()
        self.messages = [{"role": "user", "content": context["task"]}]

        for stage in self.stages:
            self._run_stage(stage)
            if self.env.state.submitted:
                break

        # Force submit if the agent never submitted
        if not self.env.state.submitted:
            self._forced_submit()

        summary = self.env.get_summary()
        summary["input_tokens"]    = self.total_input_tokens
        summary["output_tokens"]   = self.total_output_tokens
        summary["model"]           = self.model
        summary["stage_log"]       = self.stage_log
        summary["elapsed_seconds"] = round(time.time() - self._start_time, 1)
        return summary

    def _run_stage(self, stage: dict) -> None:
        stage_name = stage["name"]
        stage_label = stage["label"]
        goal = stage["goal"]
        budget = stage["budget"]

        # Announce the stage
        self.messages.append({
            "role": "user",
            "content": (
                f"[STAGE] {stage_label}\n"
                f"[GOAL] {goal}\n\n"
                f"You have up to {budget} code steps for this stage. "
                "After that you will automatically move to the next stage.\n\n"
                f"{ACTION_JSON_SCHEMA}"
            ),
        })

        stage_steps = 0
        stage_errors = 0

        while stage_steps < budget and not self.env.state.submitted:
            response = self.client.complete(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=self.messages,
            )
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
            self.messages.append({"role": "assistant", "content": response.text})

            try:
                action = Action.from_llm_response(response.text)
            except ActionParseError as exc:
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"[ERROR] Could not parse your action: {exc}\n\n"
                        f"{ACTION_JSON_SCHEMA}"
                    ),
                })
                stage_errors += 1
                continue

            observation = self.env.step(action)
            feedback = observation.to_feedback_message()

            # Inject notebook context (last 3 cells) — same as GymAgent
            notebook_ctx = self.env.state.cell_history.to_feedback_context(
                max_cells=3, max_code_chars=500, max_output_chars=250
            )
            if notebook_ctx:
                feedback = f"{feedback}\n\n{notebook_ctx}"

            # Inject stage budget so agent knows both env budget and stage budget
            stage_remaining = budget - stage_steps - 1
            if stage_remaining > 0:
                feedback += (
                    f"\n\n[STAGE BUDGET] {stage_remaining} steps remaining in {stage_label}."
                )
            else:
                feedback += f"\n\n[STAGE BUDGET] This was your last step in {stage_label}."

            self.messages.append({"role": "user", "content": feedback})

            if action.type == "code":
                stage_steps += 1
            if observation.stderr.strip():
                stage_errors += 1
            if observation.submitted or observation.done:
                break

        self.stage_log.append({
            "stage": stage_name,
            "steps": stage_steps,
            "errors": stage_errors,
            "checklist_coverage": self.env.checklist.coverage(),
        })

    def _forced_submit(self) -> None:
        """Try to submit whatever model is in the workspace."""
        ns = self.env.state.workspace.namespace
        # Prefer canonical names
        model_var = None
        for name in ["best_model", "model"]:
            if ns.get(name) is not None:
                model_var = name
                break
        # Fall back: scan for any object with predict()
        if model_var is None:
            for k, v in ns.items():
                if not k.startswith("_") and callable(getattr(v, "predict", None)):
                    model_var = k
                    break
        if model_var is None:
            return
        obs = self.env.step(Action.submit_action(model_var))
        self.messages.append({"role": "user", "content": obs.to_feedback_message()})


def _summary_metrics(summary: dict) -> dict:
    metrics = {
        "has_test_metric": int(summary.get("test_metric") is not None),
        "submit_failed": int(summary.get("test_metric") is None),
        "checklist_coverage": summary["checklist_coverage"],
        "steps_used": summary["steps_used"],
        "error_count": summary.get("error_count", summary.get("errors_count", 0)),
        "input_tokens": summary.get("input_tokens", 0),
        "output_tokens": summary.get("output_tokens", 0),
        "elapsed_seconds": summary.get("elapsed_seconds", 0),
    }
    if summary.get("test_metric") is not None:
        metrics["test_metric"] = summary["test_metric"]
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _dataset_name(splits: DatasetSplits, dataset_arg: str | None) -> str:
    if splits.metadata.name:
        return splits.metadata.name
    if dataset_arg:
        return os.path.splitext(os.path.basename(dataset_arg))[0]
    return "dataset"


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Single CSV file; requires --target")
    source.add_argument(
        "--dataset-dir",
        help="Directory with train.csv, val.csv, test.csv, meta.json",
    )
    parser.add_argument("--target", help="Target column for --dataset CSV mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--model", default=None, help="Override LLM_MODEL env var")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--sandbox-timeout", type=int, default=None)
    parser.add_argument("--experiment-name", default="autovibe-gym")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    defaults = MODE_DEFAULTS[args.mode]
    max_tokens = args.max_tokens or defaults["max_tokens"]
    sandbox_timeout = args.sandbox_timeout or defaults["sandbox_timeout"]
    stages = defaults["stages"]

    splits = load_dataset_splits(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        target_col=args.target,
        seed=args.seed,
    )
    metric_fn, metric_name = resolve_metric(splits.metadata, splits.train[splits.target_col])
    max_steps = sum(s["budget"] for s in stages) + 5  # small buffer for submit

    dataset_name = _dataset_name(splits, args.dataset_dir or args.dataset)
    model_name = args.model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    run_name = args.run_name or f"fixed_{dataset_name}_{model_name.split('/')[-1]}"

    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "mode": args.mode,
            "model": model_name,
            "dataset": dataset_name,
            "experiment_type": "fixed_transitions",
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "sandbox_timeout": sandbox_timeout,
            "n_stages": len(stages),
            "stage_budgets": json.dumps({s["name"]: s["budget"] for s in stages}),
            "dataset_suite": splits.metadata.suite or "legacy",
            "dataset_split_strategy": splits.metadata.split_strategy,
            "dataset_role": splits.metadata.role,
            "dataset_sampled": str(splits.metadata.sampled),
        })

        env = GymEnv(
            train=splits.train,
            val=splits.val,
            test=splits.test,
            target_col=splits.target_col,
            metric_fn=metric_fn,
            metric_name=metric_name,
            max_steps=max_steps,
            sandbox_timeout=sandbox_timeout,
        )

        agent = FixedTransitionsAgent(
            env=env,
            stages=stages,
            model=model_name,
            max_tokens=max_tokens,
        )
        summary = agent.run()
        mlflow.log_text(env.state.cell_history.to_markdown(), "cell_history.md")
        mlflow.log_text(
            json.dumps(summary.get("stage_log", []), indent=2),
            "stage_log.json",
        )

        mlflow.log_metrics(_summary_metrics(summary))

    print("\n=== Run Summary ===")
    stage_log = summary.pop("stage_log", [])
    print(json.dumps(summary, indent=2))
    if stage_log:
        print("\n=== Stage Log ===")
        for entry in stage_log:
            print(f"  {entry['stage']}: {entry['steps']} steps, "
                  f"{entry['errors']} errors, "
                  f"coverage={entry['checklist_coverage']:.2f}")


if __name__ == "__main__":
    main()

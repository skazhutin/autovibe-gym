"""
Fixed gym experiment runner.

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

from experiments.modes import add_mode_metadata_args, mode_metadata_params
from gym import NotebookGymEnv
from gym.agent import SYSTEM_PROMPT, THOUGHTS_DISABLED_PROMPT, THOUGHTS_ENABLED_PROMPT
from gym.datasets import DatasetSplits, load_dataset_splits, metric_from_name, resolve_metric
from gym.llm import make_llm_client
from gym.model_config import apply_model_reference
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
        "budget": 5,
    },
    {
        "name": "feature_engineering",
        "label": "Stage 3/5 — Feature Engineering",
        "goal": (
            "Create or transform features. Consider scaling, log transforms, "
            "interaction terms, or domain-specific encodings. "
            "Evaluate impact on validation score."
        ),
        "budget": 5,
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

# Cloud: reduce only EDA and tuning; keep preprocessing/feature_eng full budget
# so the agent has enough room to recover from errors before model training.
_CLOUD_BUDGET_CUTS = {"eda": 1, "hyperparameter_tuning": 1}
STAGES_CLOUD = [
    {**s, "budget": max(2, s["budget"] - _CLOUD_BUDGET_CUTS.get(s["name"], 0))}
    for s in STAGES_LOCAL
]

MODE_DEFAULTS = {
    "local": {"max_tokens": 8192, "sandbox_timeout": 60, "stages": STAGES_LOCAL},
    "cloud": {"max_tokens": 4096, "sandbox_timeout": 30, "stages": STAGES_CLOUD},
}


# ---------------------------------------------------------------------------
# Fixed gym agent
# ---------------------------------------------------------------------------

CODE_OR_NOTEBOOK_STEP_ACTIONS = {
    "code",
    "add_cell",
    "update_cell",
    "delete_cell",
    "move_cell",
    "run_cell",
    "inspect_notebook",
    "restart_and_run_all",
    "validate",
}

TOOL_ACTIONS = {
    "inspect_data",
    "profile_data",
    "list_candidates",
    "check_candidate",
    "quick_validate",
    "cleanlab_diagnose",
    "tune_hyperparameters",
    "finalize",
}

class FixedTransitionsAgent:
    """
    Wraps NotebookGymEnv and drives the agent through a fixed sequence of stages.

    At each stage transition the agent receives a new user message announcing
    the stage goal. Within a stage the feedback loop is identical to the
    directive gym (checklist hints, stdout/stderr, budget).
    """

    def __init__(
        self,
        env: NotebookGymEnv,
        stages: list[dict],
        model: str,
        max_tokens: int = 8192,
        client=None,
    ):
        self.env = env
        self.stages = stages
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or make_llm_client()
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
        max_stage_turns = budget + max(3, budget)

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

        stage_turns = 0
        stage_code_steps = 0
        stage_tool_calls = 0
        stage_errors = 0
        last_action = None
        last_error_type = None
        stop_reason = "stage_budget"

        while (
            stage_turns < max_stage_turns
            and stage_code_steps < budget
            and not self.env.state.submitted
        ):
            thoughts_on = getattr(self.env, "enable_thoughts", False)
            response = self.client.complete(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT + (
                    THOUGHTS_ENABLED_PROMPT if thoughts_on else THOUGHTS_DISABLED_PROMPT
                ),
                messages=self.messages,
            )
            stage_turns += 1
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
            self.messages.append({"role": "assistant", "content": response.text})

            try:
                action = Action.from_llm_response(response.text)
            except ActionParseError as exc:
                stage_errors += 1
                last_error_type = type(exc).__name__
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"[ERROR] Could not parse your action: {exc}\n\n"
                        f"[STAGE BUDGET] {max_stage_turns - stage_turns} turns and "
                        f"{budget - stage_code_steps} code steps remain in {stage_label}.\n\n"
                        f"{ACTION_JSON_SCHEMA}"
                    ),
                })
                continue

            observation = self.env.step(action)
            last_action = action.type
            if action.type in CODE_OR_NOTEBOOK_STEP_ACTIONS:
                stage_code_steps += 1
            elif action.type in TOOL_ACTIONS:
                stage_tool_calls += 1
            feedback = observation.to_feedback_message()

            # Inject notebook context (last 3 cells) for the Jupyter backend.
            notebook_ctx = self._notebook_context(
                max_cells=3,
                max_code_chars=500,
                max_output_chars=250,
            )
            if notebook_ctx:
                feedback = f"{feedback}\n\n{notebook_ctx}"

            # Inject stage budget so agent knows both env budget and stage budget
            code_remaining = budget - stage_code_steps
            turn_remaining = max_stage_turns - stage_turns
            if code_remaining > 0 and turn_remaining > 0:
                feedback += (
                    f"\n\n[STAGE BUDGET] {code_remaining} code steps and "
                    f"{turn_remaining} total turns remaining in {stage_label}."
                )
            else:
                feedback += f"\n\n[STAGE BUDGET] This was your last turn in {stage_label}."

            if thoughts_on:
                digest = self.env.scratchpad_digest()
                if digest:
                    feedback = f"{feedback}\n\n{digest}"

            self.messages.append({"role": "user", "content": feedback})

            if observation.stderr.strip():
                stage_errors += 1
                last_error_type = observation.stderr.strip().split(":", 1)[0][:80]
                # If more than half of stage budget consumed by errors, nudge
                # agent to fall back to a simpler approach.
                if stage_errors >= budget // 2 and code_remaining > 0:
                    self.messages.append({
                        "role": "user",
                        "content": (
                            f"[STAGE HINT] You have {stage_errors} errors so far in {stage_label}. "
                            "If you are stuck, fall back to the simplest working approach "
                            "(e.g. passthrough or basic imputer) so you can move on and train a model."
                        ),
                    })
            if observation.submitted or observation.done:
                stop_reason = "submitted" if observation.submitted else "env_done"
                break

        if not self.env.state.submitted:
            if stage_turns >= max_stage_turns:
                stop_reason = "max_stage_turns"
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"[STAGE BUDGET] This fixed stage used its maximum number of turns "
                        f"({max_stage_turns}) and is moving to the next stage."
                    ),
                })
            elif stage_code_steps >= budget:
                stop_reason = "stage_budget"

        self.stage_log.append({
            "stage": stage_name,
            "turns": stage_turns,
            "code_steps": stage_code_steps,
            "tool_calls": stage_tool_calls,
            "steps": stage_code_steps,
            "errors": stage_errors,
            "stop_reason": stop_reason,
            "last_action": last_action,
            "last_error_type": last_error_type,
            "candidate_vars_seen": self.env._candidate_var_order() if hasattr(self.env, "_candidate_var_order") else [],
            "model_check_failures": getattr(self.env, "model_check_failure_count", 0),
            "checklist_coverage": self.env.checklist.coverage(),
        })

    def _notebook_context(
        self,
        *,
        max_cells: int,
        max_code_chars: int,
        max_output_chars: int,
    ) -> str:
        notebook = getattr(self.env, "notebook", None)
        if notebook is None:
            return ""
        cells = list(getattr(notebook.notebook, "cells", []))[-max_cells:]
        if not cells:
            return ""
        parts = ["[NOTEBOOK CONTEXT] Recent cells:"]
        for cell in cells:
            cell_id = str(cell.get("id", "unknown"))
            cell_type = str(cell.get("cell_type", "code"))
            source = _clip_text(str(cell.get("source", "")), max_code_chars)
            parts.append(f"- {cell_id} ({cell_type})\n{source}")
            outputs = cell.get("outputs", []) if cell_type == "code" else []
            output_text = _clip_text(_outputs_to_text(outputs), max_output_chars)
            if output_text:
                parts.append(f"  output: {output_text}")
        return "\n".join(parts)

    def _forced_submit(self) -> None:
        """Try to finalize whatever viable model is in the notebook."""
        obs = self.env.finalize("auto")
        if obs is None:
            print(
                "[fixed] WARNING: forced finalize skipped; no trained model was found. "
                "Agent exhausted stage budgets without producing a model. test_metric will be None."
            )
            return
        self.messages.append({"role": "user", "content": obs.to_feedback_message()})


def _clip_text(text: str, max_chars: int) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _outputs_to_text(outputs: list[dict]) -> str:
    chunks: list[str] = []
    for output in outputs:
        output_type = output.get("output_type")
        if output_type == "stream":
            chunks.append(str(output.get("text", "")))
        elif output_type in {"execute_result", "display_data"}:
            data = output.get("data", {})
            chunks.append(str(data.get("text/plain", "")))
        elif output_type == "error":
            traceback = output.get("traceback") or []
            chunks.append("\n".join(str(line) for line in traceback))
    return "\n".join(chunk.strip() for chunk in chunks if chunk).strip()


def _summary_metrics(summary: dict) -> dict:
    test_metric = summary.get("test_metric")
    if test_metric is None:
        test_metric = summary.get("final_test_metric")
    metrics = {
        "has_test_metric": int(test_metric is not None),
        "submit_failed": int(bool(summary.get("submit_failed", test_metric is None))),
        "checklist_coverage": summary["checklist_coverage"],
        "steps_used": summary["steps_used"],
        "error_count": summary.get("error_count", summary.get("errors_count", 0)),
        "valid_submit": int(bool(summary.get("valid_submit"))),
        "model_check_failure_count": summary.get("model_check_failure_count", 0),
        "tool_calls_total": summary.get("tool_calls_total", 0),
        "contract_feedback_count": summary.get("contract_feedback_count", 0),
        "input_tokens": summary.get("input_tokens", 0),
        "output_tokens": summary.get("output_tokens", 0),
        "elapsed_seconds": summary.get("elapsed_seconds", 0),
    }
    if summary.get("best_validation_metric") is not None:
        metrics["best_validation_metric"] = summary["best_validation_metric"]
    if test_metric is not None:
        metrics["test_metric"] = test_metric
        metrics["final_test_metric"] = test_metric
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
    parser.add_argument("--model", required=True, help="Model id or name from the shared model registry")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override total step budget (default: sum of stage budgets + 5).",
    )
    parser.add_argument("--sandbox-timeout", type=int, default=None)
    parser.add_argument("--workspace-dir", default=None)
    parser.add_argument("--enable-thoughts", action="store_true",
                        help="Let the agent keep a persistent scratchpad of visible thoughts.")
    parser.add_argument("--experiment-name", default="autovibe-gym")
    parser.add_argument("--run-name", default=None)
    add_mode_metadata_args(parser)
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
    # Total step budget defaults to the sum of stage budgets plus a small
    # buffer for the submit action; --max-steps overrides it for CLI parity
    # with the other run_* scripts and the batch matrix runner.
    max_steps = args.max_steps or (sum(s["budget"] for s in stages) + 5)

    dataset_name = _dataset_name(splits, args.dataset_dir or args.dataset)
    model_name = apply_model_reference(args.model)
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
            "experiment_type": "fixed_gym",
            "thoughts_enabled": args.enable_thoughts,
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "sandbox_timeout": sandbox_timeout,
            "n_stages": len(stages),
            "stage_budgets": json.dumps({s["name"]: s["budget"] for s in stages}),
            "dataset_split_strategy": splits.metadata.split_strategy,
            "dataset_role": splits.metadata.role,
            "dataset_sampled": str(splits.metadata.sampled),
            **mode_metadata_params(args, "fixed_gym"),
        })

        env = NotebookGymEnv(
            train=splits.train,
            val=splits.val,
            test=splits.test,
            target_col=splits.target_col,
            metric_fn=metric_fn,
            metric_name=metric_name,
            max_steps=max_steps,
            workspace_dir=args.workspace_dir,
            mode="directive_gym",
            kernel_timeout=sandbox_timeout,
            enable_thoughts=args.enable_thoughts,
        )

        agent = FixedTransitionsAgent(
            env=env,
            stages=stages,
            model=model_name,
            max_tokens=max_tokens,
        )
        summary = agent.run()
        summary.update(mode_metadata_params(args, "fixed_gym"))
        episode_workspace = summary.get("episode_workspace")

        # Best-effort post-run self-summary, once the agent has solved the task
        # (reached a final submit — even if the hidden test later rejected it),
        # persisted as run_summary.json so the dashboard «Мысли» tab shows it
        # above the step-by-step thoughts.
        if summary.get("submitted"):
            from gym.run_summary import generate_and_write, read_solution_code

            generate_and_write(
                agent.client,
                agent.model,
                episode_workspace or args.workspace_dir,
                conversation=agent.messages,
                solution_code=read_solution_code(episode_workspace or args.workspace_dir),
                max_tokens=min(max_tokens, 700),
            )

        if episode_workspace and os.path.isdir(episode_workspace):
            mlflow.log_artifacts(episode_workspace, artifact_path="episode")
        private_episode_dir = summary.get("private_episode_dir")
        if private_episode_dir and os.path.isdir(private_episode_dir):
            mlflow.log_artifacts(private_episode_dir, artifact_path="private_episode")
        mlflow.log_text(
            json.dumps(summary.get("stage_log", []), indent=2),
            "stage_log.json",
        )

        mlflow.log_metrics(_summary_metrics(summary))
        mlflow.set_tags({
            "final_status": summary.get("final_status") or "",
            "null_reason": summary.get("null_reason") or "",
            "finalize_path": summary.get("finalize_path") or "",
        })
        env.close()

    print("\n=== Run Summary ===")
    stage_log = summary.pop("stage_log", [])
    print(json.dumps(summary, indent=2))
    if stage_log:
        print("\n=== Stage Log ===")
        for entry in stage_log:
            print(f"  {entry['stage']}: {entry['steps']} code steps, "
                  f"{entry.get('tool_calls', 0)} tool calls, "
                  f"{entry.get('turns', entry['steps'])} turns, "
                  f"{entry['errors']} errors, "
                  f"coverage={entry['checklist_coverage']:.2f}")


if __name__ == "__main__":
    main()

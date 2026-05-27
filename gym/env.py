import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .checklist import Checklist
from .executor import CodeExecutor
from .protocol import Action, Observation, StepResult, coerce_action
from .workspace import Workspace


@dataclass
class EnvState:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    target_col: str
    metric_name: str
    workspace: Workspace
    step: int = 0
    max_steps: int = 20
    submitted: bool = False
    history: list[Observation] = field(default_factory=list)

    @property
    def namespace(self) -> dict[str, Any]:
        return self.workspace.namespace

    @namespace.setter
    def namespace(self, namespace: dict[str, Any]) -> None:
        self.workspace.update_namespace(namespace)


class GymEnv:
    """
    Iterative ML environment for LLM agents.

    The agent sends explicit actions. Code actions run inside a persistent
    workspace namespace; submit actions evaluate one named model on hidden test.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        target_col: str,
        metric_fn: Callable,
        metric_name: str = "score",
        max_steps: int = 20,
        sandbox_timeout: int = 60,
    ):
        workspace = Workspace(train=train, val=val, target_col=target_col)
        self.state = EnvState(
            train=train,
            val=val,
            test=test,
            target_col=target_col,
            metric_name=metric_name,
            max_steps=max_steps,
            workspace=workspace,
        )
        self.metric_fn = metric_fn
        self.executor = CodeExecutor(timeout=sandbox_timeout)
        self.checklist = Checklist(target_col=target_col)
        self._start_time = time.time()

    def reset(self) -> dict:
        self.state.step = 0
        self.state.submitted = False
        self.state.history = []
        self.state.workspace.reset()
        self.checklist = Checklist(target_col=self.state.target_col)
        self._start_time = time.time()
        return self._build_context_prompt()

    def step(self, action: Action | dict[str, Any] | str) -> Observation:
        """Execute one agent action and return structured feedback."""
        if self.state.submitted:
            raise RuntimeError("Environment already finalized via submit().")

        parsed = coerce_action(action)
        if parsed.type == "submit":
            return self.submit_by_name(parsed.model_var)

        return self._run_code(parsed)

    def submit_by_name(self, model_var: str = "model") -> Observation:
        """Submit a model stored in the workspace by variable name."""
        model = self.state.workspace.get(model_var)
        if model is None:
            observation = Observation(
                action="submit",
                step=self.state.step,
                budget_remaining=self.budget_remaining(),
                stderr=(
                    f"No variable named '{model_var}' found in workspace. "
                    "Train a model and assign it before submitting."
                ),
                checklist_coverage=self.checklist.coverage(),
                model_var=model_var,
            )
            self.state.history.append(observation)
            return observation

        self.submit(model, model_var=model_var)
        return self.state.history[-1]

    def submit(self, model: Any, model_var: str | None = None) -> float:
        """One-shot final evaluation on hidden test set. Closes the environment."""
        if self.state.submitted:
            raise RuntimeError("Already submitted.")
        self.state.submitted = True

        X_test = self.state.test.drop(columns=[self.state.target_col])
        y_test = self.state.test[self.state.target_col]
        preds = model.predict(X_test)
        score = float(self.metric_fn(y_test, preds))

        observation = Observation(
            action="submit",
            step=self.state.step,
            budget_remaining=self.budget_remaining(),
            stdout=f"{self.state.metric_name}={score:.4f}",
            checklist_coverage=self.checklist.coverage(),
            done=True,
            submitted=True,
            test_metric=score,
            model_var=model_var,
        )
        self.state.history.append(observation)
        return score

    def budget_remaining(self) -> int:
        return max(self.state.max_steps - self.state.step, 0)

    def _run_code(self, action: Action) -> Observation:
        if self.state.step >= self.state.max_steps:
            observation = Observation(
                action="code",
                code=action.code,
                step=self.state.step,
                budget_remaining=0,
                stderr="Step budget exhausted. Submit an existing model instead.",
                checklist_coverage=self.checklist.coverage(),
                done=True,
            )
            self.state.history.append(observation)
            return observation

        self.state.step += 1
        stdout, stderr, namespace = self.executor.run(
            action.code, namespace=self.state.namespace
        )
        self.state.namespace = namespace

        hints = self.checklist.evaluate(
            code=action.code,
            stdout=stdout,
            namespace=self.state.namespace,
            history=self.state.history,
        )
        coverage = self.checklist.coverage()

        observation = Observation(
            action="code",
            code=action.code,
            stdout=stdout,
            stderr=stderr,
            hints=hints,
            checklist_coverage=coverage,
            step=self.state.step,
            budget_remaining=self.budget_remaining(),
            done=self.state.step >= self.state.max_steps,
        )
        self.state.history.append(observation)
        return observation

    def _build_context_prompt(self) -> dict:
        train_info = self.state.train.describe(include="all").to_string()
        visible_symbols = ", ".join(self.state.workspace.visible_symbols())
        return {
            "task": (
                "You are solving a supervised ML task in an iterative AutoML Gym.\n"
                f"Target column: '{self.state.target_col}'\n"
                f"Metric: {self.state.metric_name}\n"
                f"Max code steps: {self.state.max_steps}\n\n"
                f"Training data shape: {self.state.train.shape}\n"
                f"Validation data shape: {self.state.val.shape}\n\n"
                f"Dataset statistics:\n{train_info}\n\n"
                "Workspace variables available to your code:\n"
                f"  {visible_symbols}\n"
                "Hidden variables:\n"
                "  test_df is never available to code actions.\n\n"
                "Each response must be one JSON action:\n"
                '  {"type": "code", "code": "print(train_df.shape)"}\n'
                '  {"type": "submit", "model_var": "best_model"}\n\n'
                "Use code actions to train and compare models. Submit exactly once "
                "when your best model is assigned to a workspace variable."
            )
        }

    def get_summary(self) -> dict:
        final = next(
            (r for r in reversed(self.state.history) if r.test_metric is not None),
            None,
        )
        error_count = sum(1 for r in self.state.history if r.stderr.strip())
        return {
            "steps_used": self.state.step,
            "checklist_coverage": self.checklist.coverage(),
            "error_count": error_count,
            "errors_count": error_count,
            "submitted": final is not None,
            "test_metric": final.test_metric if final else None,
            "elapsed_seconds": round(time.time() - self._start_time, 1),
        }

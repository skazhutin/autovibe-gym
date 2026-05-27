import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .checklist import Checklist
from .executor import CodeExecutor


@dataclass
class StepResult:
    action: str
    code: str
    stdout: str
    stderr: str
    hints: list[str]
    checklist_coverage: float
    step: int
    done: bool = False
    test_metric: float | None = None


@dataclass
class EnvState:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    target_col: str
    metric_name: str
    step: int = 0
    max_steps: int = 20
    submitted: bool = False
    history: list[StepResult] = field(default_factory=list)
    namespace: dict = field(default_factory=dict)


class GymEnv:
    """
    Iterative ML environment for LLM agents.
    The LLM submits Python code; the env executes it, runs checklist checks,
    and returns structured feedback + implicit hints.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        target_col: str,
        metric_fn,
        metric_name: str = "score",
        max_steps: int = 20,
    ):
        self.state = EnvState(
            train=train,
            val=val,
            test=test,
            target_col=target_col,
            metric_name=metric_name,
            max_steps=max_steps,
        )
        self.metric_fn = metric_fn
        self.executor = CodeExecutor()
        self.checklist = Checklist(target_col=target_col)
        self._start_time = time.time()

    def reset(self) -> dict:
        self.state.step = 0
        self.state.submitted = False
        self.state.history = []
        self.state.namespace = {}
        return self._build_context_prompt()

    def step(self, code: str) -> StepResult:
        """Execute one agent action and return structured feedback."""
        if self.state.submitted:
            raise RuntimeError("Environment already finalized via submit().")

        self.state.step += 1
        stdout, stderr, self.state.namespace = self.executor.run(
            code, namespace=self.state.namespace
        )

        hints = self.checklist.evaluate(
            code=code,
            stdout=stdout,
            namespace=self.state.namespace,
            history=self.state.history,
        )
        coverage = self.checklist.coverage()

        result = StepResult(
            action="code",
            code=code,
            stdout=stdout,
            stderr=stderr,
            hints=hints,
            checklist_coverage=coverage,
            step=self.state.step,
            done=self.state.step >= self.state.max_steps,
        )
        self.state.history.append(result)
        return result

    def submit(self, model) -> float:
        """One-shot final evaluation on test set. Closes the environment."""
        if self.state.submitted:
            raise RuntimeError("Already submitted.")
        self.state.submitted = True

        X_test = self.state.test.drop(columns=[self.state.target_col])
        y_test = self.state.test[self.state.target_col]
        preds = model.predict(X_test)
        score = self.metric_fn(y_test, preds)

        result = StepResult(
            action="submit",
            code="",
            stdout=f"{self.state.metric_name}={score:.4f}",
            stderr="",
            hints=[],
            checklist_coverage=self.checklist.coverage(),
            step=self.state.step,
            done=True,
            test_metric=score,
        )
        self.state.history.append(result)
        return score

    def budget_remaining(self) -> int:
        return self.state.max_steps - self.state.step

    def _build_context_prompt(self) -> dict:
        train_info = self.state.train.describe(include="all").to_string()
        return {
            "task": (
                f"You are solving a supervised ML task.\n"
                f"Target column: '{self.state.target_col}'\n"
                f"Metric: {self.state.metric_name}\n"
                f"Max steps: {self.state.max_steps}\n\n"
                f"Training data shape: {self.state.train.shape}\n"
                f"Validation data shape: {self.state.val.shape}\n\n"
                f"Dataset statistics:\n{train_info}\n\n"
                "Variables available in your namespace:\n"
                "  train_df, val_df  — DataFrames (test_df is hidden until submit)\n"
                "  target_col        — name of the target column (string)\n\n"
                "When ready, call env.submit(model) with your best trained model."
            )
        }

    def get_summary(self) -> dict:
        final = next(
            (r for r in reversed(self.state.history) if r.test_metric is not None),
            None,
        )
        return {
            "steps_used": self.state.step,
            "checklist_coverage": self.checklist.coverage(),
            "errors_count": sum(1 for r in self.state.history if r.stderr.strip()),
            "test_metric": final.test_metric if final else None,
            "elapsed_seconds": round(time.time() - self._start_time, 1),
        }

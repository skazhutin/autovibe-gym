import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .cell_history import CellHistory
from .checklist import Checklist
from .executor import CodeExecutor
from .protocol import Action, Observation, StepResult, coerce_action
from .workspace import Workspace

MODEL_VALIDATION_HINT = (
    "[MODEL CHECK] Workspace variable '{model_var}' cannot predict raw validation "
    "features: {error_type}: {error}. Keep preprocessing attached to the estimator "
    "so model.predict(raw validation/test features) works at submit time."
)

MODEL_INTERFACE_HINT = (
    "[MODEL CHECK] Workspace variable '{model_var}' is not submit-ready because "
    "it does not provide a predict(X) method."
)

HIDDEN_TEST_SUBMIT_ERROR = (
    "Submit failed on the hidden test split. The selected model could not predict "
    "raw hidden features. Ensure preprocessing is inside the submitted model or "
    "pipeline and handles unseen validation/test values."
)


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
    cell_history: CellHistory = field(default_factory=CellHistory)

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
        model_validation_rows: int = 32,
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
        self.model_validation_rows = model_validation_rows
        self._start_time = time.time()

    def reset(self) -> dict:
        self.state.step = 0
        self.state.submitted = False
        self.state.history = []
        self.state.cell_history.reset()
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
            return self._record_observation(observation)

        validation_hint = self._raw_validation_hint(model, model_var)
        if validation_hint:
            observation = Observation(
                action="submit",
                step=self.state.step,
                budget_remaining=self.budget_remaining(),
                stderr=validation_hint,
                hints=[validation_hint],
                checklist_coverage=self.checklist.coverage(),
                model_var=model_var,
            )
            return self._record_observation(observation)

        try:
            self.submit(model, model_var=model_var)
        except Exception:
            return self._submit_failure_observation(model_var)
        return self.state.history[-1]

    def submit(self, model: Any, model_var: str | None = None) -> float:
        """One-shot final evaluation on hidden test set. Closes the environment."""
        if self.state.submitted:
            raise RuntimeError("Already submitted.")

        X_test = self.state.test.drop(columns=[self.state.target_col])
        y_test = self.state.test[self.state.target_col]
        preds = model.predict(X_test)

        # Coerce predictions to match y_test dtype to handle label-encoding mismatches
        # (e.g. agent encodes strings→ints but test retains original string labels)
        try:
            score = float(self.metric_fn(y_test, preds))
        except (ValueError, TypeError):
            import numpy as np
            import pandas as _pd
            try:
                preds_coerced = _pd.Series(preds).astype(y_test.dtype).values
                score = float(self.metric_fn(y_test, preds_coerced))
            except Exception:
                classes = sorted(y_test.unique())
                try:
                    preds_mapped = np.array([classes[int(p)] for p in preds])
                    score = float(self.metric_fn(y_test, preds_mapped))
                except Exception as e:
                    raise ValueError(
                        f"Cannot evaluate predictions: {e}. "
                        "Ensure model.predict() returns labels matching "
                        f"the target column (e.g. {list(y_test.unique()[:3])})."
                    ) from e

        self.state.submitted = True
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
        self._record_observation(observation)
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
            return self._record_observation(observation)

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
        hints.extend(self._model_validation_hints())
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
        return self._record_observation(observation)

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
                "Notebook workflow:\n"
                "  Each code action is stored as a new notebook-like cell.\n"
                "  Workspace variables persist across cells, so build on prior work "
                "instead of rewriting everything from scratch.\n\n"
                "Model readiness check:\n"
                "  When model or best_model exists, the environment checks whether "
                "it can predict raw validation features. Fix any [MODEL CHECK] "
                "feedback before submitting.\n\n"
                "Each response must be one JSON action:\n"
                '  {"type": "code", "code": "print(train_df.shape)"}\n'
                '  {"type": "submit", "model_var": "best_model"}\n\n'
                "Use code actions to train and compare models. Submit exactly once "
                "when your best model is assigned to a workspace variable."
            )
        }

    def _record_observation(self, observation: Observation) -> Observation:
        self.state.history.append(observation)
        self.state.cell_history.append_observation(observation)
        return observation

    def _model_validation_hints(self) -> list[str]:
        hints = []
        seen_ids = set()
        for model_var in ("best_model", "model"):
            model = self.state.workspace.get(model_var)
            if model is None:
                continue
            if id(model) in seen_ids:
                continue
            seen_ids.add(id(model))
            hint = self._raw_validation_hint(model, model_var)
            if hint:
                hints.append(hint)
        return hints

    def _raw_validation_hint(self, model: Any, model_var: str) -> str | None:
        if not hasattr(model, "predict"):
            return MODEL_INTERFACE_HINT.format(model_var=model_var)

        X_val = self.state.val.drop(columns=[self.state.target_col])
        if X_val.empty:
            return None
        if self.model_validation_rows > 0:
            X_val = X_val.head(self.model_validation_rows)

        try:
            model.predict(X_val)
        except Exception as exc:
            return MODEL_VALIDATION_HINT.format(
                model_var=model_var,
                error_type=type(exc).__name__,
                error=_clip(str(exc), 240),
            )
        return None

    def _submit_failure_observation(self, model_var: str | None) -> Observation:
        self.state.submitted = True
        observation = Observation(
            action="submit",
            step=self.state.step,
            budget_remaining=self.budget_remaining(),
            stderr=HIDDEN_TEST_SUBMIT_ERROR,
            checklist_coverage=self.checklist.coverage(),
            done=True,
            submitted=True,
            test_metric=None,
            model_var=model_var,
        )
        return self._record_observation(observation)

    def get_summary(self) -> dict:
        final = next(
            (r for r in reversed(self.state.history) if r.submitted),
            None,
        )
        error_count = sum(1 for r in self.state.history if r.stderr.strip())
        return {
            "steps_used": self.state.step,
            "checklist_coverage": self.checklist.coverage(),
            "error_count": error_count,
            "errors_count": error_count,
            "cells_used": len(self.state.cell_history),
            "submitted": self.state.submitted or final is not None,
            "test_metric": final.test_metric if final else None,
            "elapsed_seconds": round(time.time() - self._start_time, 1),
        }


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "... [truncated]"

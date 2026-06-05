import time
from dataclasses import dataclass, field
from typing import Any, Callable
import inspect

import pandas as pd

from .cell_history import CellHistory
from .checklist import Checklist
from .data_profile import build_dataset_card, format_profile_for_agent, build_compact_profile
from .executor import CodeExecutor
from .protocol import AGENT_STAGE_VALUES, Action, ActionParseError, Observation, StepResult, coerce_action
from .workspace import Workspace

MODEL_VALIDATION_HINT = (
    "[MODEL CHECK] Workspace variable '{model_var}' cannot predict raw validation "
    "features: {error_type}: {error}. The saved candidate must reproduce all "
    "required preprocessing when called on new raw rows."
)

MODEL_INTERFACE_HINT = (
    "[MODEL CHECK] Workspace variable '{model_var}' is not submit-ready because "
    "it does not provide a predict(X) method."
)

HIDDEN_TEST_SUBMIT_ERROR = (
    "Submit failed on the hidden test split. The selected model could not predict "
    "raw hidden features. Ensure preprocessing is inside the submitted model or "
    "pipeline and handles unseen validation/test values. Error type: {error_type}."
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
        executor_backend: str | None = None,
        sandbox_image: str | None = None,
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
        self.executor = CodeExecutor(
            timeout=sandbox_timeout,
            backend=executor_backend,
            docker_image=sandbox_image,
        )
        self.checklist = Checklist(target_col=target_col)
        self.model_validation_rows = model_validation_rows
        self.model_check_failure_count = 0
        self.final_status: str | None = None
        self.null_reason: str | None = None
        self._start_time = time.time()
        self.enable_thoughts = False
        self.current_stage: str | None = None

    def reset(self) -> dict:
        self.state.step = 0
        self.state.submitted = False
        self.state.history = []
        self.state.cell_history.reset()
        self.state.workspace.reset()
        self.checklist = Checklist(target_col=self.state.target_col)
        self.model_check_failure_count = 0
        self.final_status = None
        self.null_reason = None
        self._start_time = time.time()
        self.current_stage = None
        return self._build_context_prompt()

    def step(self, action: Action | dict[str, Any] | str) -> Observation:
        """Execute one agent action and return structured feedback."""
        if self.state.submitted:
            raise RuntimeError("Environment already finalized via submit().")

        try:
            parsed = coerce_action(action)
        except ActionParseError as exc:
            return self._contract_observation("invalid_action", str(exc), action_type="invalid_action")
        contract_error = self._validate_action_contract(parsed)
        if contract_error is not None:
            key, message = contract_error
            return self._contract_observation(
                key,
                message,
                action_type=parsed.type,
                stage=parsed.stage or None,
                thoughts=parsed.thoughts or None,
                model_var=parsed.model_var if parsed.type == "validate" else None,
            )
        self.current_stage = parsed.stage
        if parsed.type == "submit":
            return self.submit_by_name(parsed.model_var)
        if parsed.type == "validate":
            return self._check_or_quick_validate(parsed.model_var, quick=True)
        if parsed.type == "finalize":
            return self.submit_by_name(parsed.model_var)
        if parsed.type == "inspect_data":
            card = build_dataset_card(
                self.state.train,
                self.state.val,
                self.state.target_col,
                self.state.metric_name,
            )
            return self._record_observation(
                Observation(
                    action="inspect_data",
                    step=self.state.step,
                    budget_remaining=self.budget_remaining(),
                    stdout=card,
                    checklist_coverage=self.checklist.coverage(),
                )
            )
        if parsed.type == "profile_data":
            profile = build_compact_profile(
                self.state.train,
                self.state.val,
                self.state.target_col,
                self.state.metric_name,
            )
            return self._record_observation(
                Observation(
                    action="profile_data",
                    step=self.state.step,
                    budget_remaining=self.budget_remaining(),
                    stdout=format_profile_for_agent(profile),
                    checklist_coverage=self.checklist.coverage(),
                )
            )
        if parsed.type == "list_candidates":
            names = self._candidate_var_order()
            text = "[CANDIDATES]\n" + ("\n".join(f"- {n}" for n in names) if names else "No predict-capable candidate variables were discovered.")
            return self._record_observation(
                Observation(
                    action="list_candidates",
                    step=self.state.step,
                    budget_remaining=self.budget_remaining(),
                    stdout=text,
                    checklist_coverage=self.checklist.coverage(),
                )
            )
        if parsed.type in {"check_candidate", "quick_validate"}:
            return self._check_or_quick_validate(parsed.model_var, quick=parsed.type == "quick_validate")
        if parsed.type in {"cleanlab_diagnose", "tune_hyperparameters"}:
            return self._record_observation(
                Observation(
                    action=parsed.type,
                    step=self.state.step,
                    budget_remaining=self.budget_remaining(),
                    stage=parsed.stage,
                    stdout=f"[{parsed.type.upper()}] This tool is available in NotebookGymEnv.",
                    checklist_coverage=self.checklist.coverage(),
                    model_var=parsed.model_var,
                )
            )

        return self._run_code(parsed)

    def _validate_action_contract(self, action: Action) -> tuple[str, str] | None:
        if not action.stage:
            return (
                "missing_stage",
                "Every JSON action must include a non-empty 'stage' from the allowed stage enum.",
            )
        if action.stage not in AGENT_STAGE_VALUES:
            return (
                "unknown_stage",
                f"Unknown stage '{action.stage}'. Use one of the allowed stage values.",
            )
        if action.type == "think":
            return ("think_disabled", "Thoughts mode is disabled, so type 'think' is not allowed.")
        if action.stage == "planning":
            return ("planning_disabled", "Thoughts mode is disabled, so stage 'planning' is not allowed.")
        if action.thoughts:
            return ("thoughts_disabled", "Thoughts mode is disabled, so do not include 'thoughts'.")
        return None

    def _contract_observation(
        self,
        key: str,
        message: str,
        *,
        action_type: str,
        stage: str | None = None,
        thoughts: str | None = None,
        model_var: str | None = None,
    ) -> Observation:
        del key
        return self._record_observation(
            Observation(
                action=action_type,
                step=self.state.step,
                budget_remaining=self.budget_remaining(),
                stage=stage,
                thoughts=thoughts,
                stderr=message,
                checklist_coverage=self.checklist.coverage(),
                model_var=model_var,
            )
        )

    def submit_by_name(self, model_var: str = "model") -> Observation:
        """Submit a model stored in the workspace by variable name."""
        if model_var == "auto":
            model_var = self._candidate_var_order()[0] if self._candidate_var_order() else "model"
        model = self.state.workspace.get(model_var)
        if model is None:
            self.final_status = "no_candidate_found"
            self.null_reason = f"No variable named '{model_var}' found in workspace."
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
            self.final_status = "submit_blocked_preflight"
            self.null_reason = validation_hint
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
        except Exception as exc:
            return self._submit_failure_observation(model_var, exc)
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
        self.final_status = "submitted_clean"
        self.null_reason = None
        observation = Observation(
            action="submit",
            step=self.state.step,
            budget_remaining=self.budget_remaining(),
            stdout="[SUBMITTED] Final candidate accepted. Episode finished.",
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
        visible_symbols = ", ".join(self.state.workspace.visible_symbols())
        dataset_card = build_dataset_card(
            self.state.train,
            self.state.val,
            self.state.target_col,
            self.state.metric_name,
            max_chars=4500,
        )
        return {
            "task": (
                "You are solving a supervised ML task in an iterative AutoML Gym.\n"
                f"Target column: '{self.state.target_col}'\n"
                f"Metric: {self.state.metric_name}\n"
                f"Max code steps: {self.state.max_steps}\n\n"
                f"{dataset_card}\n\n"
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
                '  {"type": "code", "stage": "data_schema_inspection", "code": "print(train_df.shape)"}\n'
                '  {"type": "submit", "stage": "submission", "model_var": "best_model"}\n\n'
                "Every action must include a non-planning stage. Thoughts mode is disabled here: "
                "do not include thoughts, do not use type think, and do not use stage planning.\n\n"
                "Use code actions to train and compare models. Submit exactly once "
                "when your best model is assigned to a workspace variable."
            )
        }

    def _record_observation(self, observation: Observation) -> Observation:
        if observation.stage is None:
            observation.stage = self.current_stage
        self.state.history.append(observation)
        self.state.cell_history.append_observation(observation)
        return observation

    def _model_validation_hints(self) -> list[str]:
        hints = []
        seen_ids = set()
        for model_var in self._candidate_var_order() or ["best_model", "model"]:
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

    def _candidate_var_order(self) -> list[str]:
        priority = [
            "model",
            "best_model",
            "final_model",
            "pipeline",
            "best_pipeline",
            "best_estimator",
            "estimator",
            "clf",
            "classifier",
            "best_clf",
            "best_classifier",
            "rf_model",
            "lgb_model",
            "xgb_model",
            "catboost_model",
            "trained_model",
            "final_pipeline_model",
            "reg",
            "regressor",
        ]
        discovered = [
            name
            for name, value in self.state.workspace.namespace.items()
            if not name.startswith("_")
            and not inspect.isclass(value)
            and callable(getattr(value, "predict", None))
        ]
        ordered: list[str] = []
        for name in priority + discovered:
            if name in self.state.workspace.namespace and name not in ordered:
                ordered.append(name)
        return ordered

    def _check_or_quick_validate(self, model_var: str, *, quick: bool) -> Observation:
        names = self._candidate_var_order() if model_var == "auto" else [model_var]
        lines = ["[QUICK VALIDATE]" if quick else "[CANDIDATE CHECK]"]
        selected_metric = None
        selected_name = model_var
        for name in names:
            model = self.state.workspace.get(name)
            if model is None:
                lines.append(f"- {name}: exists=false")
                continue
            hint = self._raw_validation_hint(model, name)
            if hint:
                lines.append(f"- {name}: raw_validation_ready=false issue={hint}")
                continue
            if quick:
                try:
                    X_val = self.state.val.drop(columns=[self.state.target_col])
                    y_val = self.state.val[self.state.target_col]
                    metric = float(self.metric_fn(y_val, model.predict(X_val)))
                    selected_metric = metric
                    selected_name = name
                    lines.append(f"- {name}: validation_{self.state.metric_name}={metric:.6f}")
                    break
                except Exception as exc:
                    lines.append(f"- {name}: validation_failed={type(exc).__name__}: {_clip(str(exc), 200)}")
            else:
                selected_name = name
                lines.append(f"- {name}: raw_validation_ready=true serializable=not_checked")
        return self._record_observation(
            Observation(
                action="quick_validate" if quick else "check_candidate",
                step=self.state.step,
                budget_remaining=self.budget_remaining(),
                stdout="\n".join(lines),
                checklist_coverage=self.checklist.coverage(),
                validation_metric=selected_metric,
                model_var=selected_name,
            )
        )

    def _raw_validation_hint(self, model: Any, model_var: str) -> str | None:
        if not hasattr(model, "predict"):
            self.model_check_failure_count += 1
            return MODEL_INTERFACE_HINT.format(model_var=model_var)

        X_val = self.state.val.drop(columns=[self.state.target_col])
        if X_val.empty:
            return None
        if self.model_validation_rows > 0:
            X_val = X_val.head(self.model_validation_rows)

        try:
            model.predict(X_val)
        except Exception as exc:
            self.model_check_failure_count += 1
            return MODEL_VALIDATION_HINT.format(
                model_var=model_var,
                error_type=type(exc).__name__,
                error=_clip(str(exc), 240),
            )
        return None

    def _submit_failure_observation(
        self,
        model_var: str | None,
        exc: Exception,
    ) -> Observation:
        self.state.submitted = True
        self.final_status = "hidden_submit_failed"
        self.null_reason = "Candidate passed validation preflight but failed on hidden test."
        observation = Observation(
            action="submit",
            step=self.state.step,
            budget_remaining=self.budget_remaining(),
            stderr=HIDDEN_TEST_SUBMIT_ERROR.format(
                error_type=type(exc).__name__,
            ),
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
            "valid_submit": bool(final and final.test_metric is not None),
            "final_status": self.final_status,
            "null_reason": self.null_reason,
            "model_check_failure_count": self.model_check_failure_count,
            "elapsed_seconds": round(time.time() - self._start_time, 1),
        }


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "... [truncated]"

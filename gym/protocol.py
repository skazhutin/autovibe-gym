import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal


ActionType = Literal[
    "code",
    "add_cell",
    "update_cell",
    "delete_cell",
    "move_cell",
    "run_cell",
    "inspect_notebook",
    "restart_and_run_all",
    "validate",
    "submit",
]


class ActionParseError(ValueError):
    """Raised when an LLM response looks like an action but is invalid."""


ACTION_JSON_SCHEMA = """
Respond with exactly one JSON object.

Code action:
{"type": "code", "code": "print(train_df.shape)"}

Add code cell action:
{"type": "add_cell", "cell_type": "code", "source": "print(train_df.shape)", "execute": true}

Add markdown cell action:
{"type": "add_cell", "cell_type": "markdown", "source": "## Data exploration", "execute": false}

Update cell action:
{"type": "update_cell", "cell_id": "cell_03", "source": "print(val_df.shape)", "execute": true}

Run, move, delete, or inspect:
{"type": "run_cell", "cell_id": "cell_03"}
{"type": "move_cell", "cell_id": "cell_03", "new_position": 1}
{"type": "delete_cell", "cell_id": "cell_03"}
{"type": "inspect_notebook"}

Clean reproducibility action:
{"type": "restart_and_run_all"}

Validate action:
{"type": "validate", "model_var": "model"}

Submit action:
{"type": "submit", "model_var": "model"}
""".strip()


@dataclass(frozen=True)
class Action:
    """One explicit agent action sent to the environment."""

    type: ActionType
    code: str = ""
    cell_type: str = "code"
    source: str = ""
    execute: bool = False
    cell_id: str | None = None
    new_position: int | None = None
    model_var: str = "model"

    def __post_init__(self) -> None:
        if self.type not in {
            "code",
            "add_cell",
            "update_cell",
            "delete_cell",
            "move_cell",
            "run_cell",
            "inspect_notebook",
            "restart_and_run_all",
            "validate",
            "submit",
        }:
            raise ActionParseError(f"Unsupported action type: {self.type!r}")
        if self.type == "code" and not isinstance(self.code, str):
            raise ActionParseError("Code action requires a string 'code' field.")
        if self.type == "add_cell":
            if self.cell_type not in {"code", "markdown"}:
                raise ActionParseError("add_cell requires cell_type 'code' or 'markdown'.")
            if not isinstance(self.source, str):
                raise ActionParseError("add_cell requires a string 'source' field.")
        if self.type == "update_cell":
            if not self.cell_id:
                raise ActionParseError("update_cell requires a non-empty 'cell_id'.")
            if not isinstance(self.source, str):
                raise ActionParseError("update_cell requires a string 'source' field.")
        if self.type in {"delete_cell", "run_cell"} and not self.cell_id:
            raise ActionParseError(f"{self.type} requires a non-empty 'cell_id'.")
        if self.type == "move_cell":
            if not self.cell_id:
                raise ActionParseError("move_cell requires a non-empty 'cell_id'.")
            if self.new_position is None:
                raise ActionParseError("move_cell requires a 'new_position'.")
        if self.type in {"validate", "submit"} and not self.model_var:
            raise ActionParseError(f"{self.type} requires a non-empty 'model_var'.")

    @classmethod
    def code_action(cls, code: str) -> "Action":
        return cls(type="code", code=code)

    @classmethod
    def submit_action(cls, model_var: str = "model") -> "Action":
        return cls(type="submit", model_var=model_var)

    @classmethod
    def add_cell_action(
        cls,
        source: str,
        *,
        cell_type: str = "code",
        execute: bool = False,
    ) -> "Action":
        return cls(
            type="add_cell",
            cell_type=cell_type,
            source=source,
            execute=execute,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Action":
        raw_type = payload.get("type") or payload.get("action")
        action_type = _normalize_action_type(raw_type)

        if action_type == "code":
            code = payload.get("code")
            if code is None:
                raise ActionParseError("Code action is missing the 'code' field.")
            return cls.code_action(str(code))

        if action_type == "add_cell":
            return cls.add_cell_action(
                str(payload.get("source") or ""),
                cell_type=str(payload.get("cell_type") or "code"),
                execute=bool(payload.get("execute", False)),
            )

        if action_type == "update_cell":
            return cls(
                type="update_cell",
                cell_id=str(payload.get("cell_id") or ""),
                source=str(payload.get("source") or ""),
                execute=bool(payload.get("execute", False)),
            )

        if action_type in {"delete_cell", "run_cell"}:
            return cls(type=action_type, cell_id=str(payload.get("cell_id") or ""))

        if action_type == "move_cell":
            try:
                new_position = int(payload.get("new_position"))
            except (TypeError, ValueError) as exc:
                raise ActionParseError("move_cell requires an integer 'new_position'.") from exc
            return cls(
                type="move_cell",
                cell_id=str(payload.get("cell_id") or ""),
                new_position=new_position,
            )

        if action_type in {"inspect_notebook", "restart_and_run_all"}:
            return cls(type=action_type)

        if action_type == "validate":
            return cls(type="validate", model_var=str(payload.get("model_var") or "model"))

        if action_type == "submit":
            return cls.submit_action(str(payload.get("model_var") or "model"))

        raise ActionParseError(f"Unsupported action type: {raw_type!r}")

    @classmethod
    def from_json(cls, text: str) -> "Action":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"Invalid action JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ActionParseError("Action JSON must be an object.")
        return cls.from_payload(payload)

    @classmethod
    def from_llm_response(cls, text: str) -> "Action":
        """Parse the new JSON protocol, with legacy code-output fallback."""
        json_text = _extract_json_block(text)
        if json_text is not None:
            return cls.from_json(json_text)

        stripped = text.strip()
        if stripped.upper() == "SUBMIT":
            return cls.submit_action("model")

        return cls.code_action(_extract_code_block(text))

    def to_dict(self) -> dict[str, Any]:
        if self.type == "code":
            return {"type": "code", "code": self.code}
        if self.type == "add_cell":
            return {
                "type": self.type,
                "cell_type": self.cell_type,
                "source": self.source,
                "execute": self.execute,
            }
        if self.type == "update_cell":
            return {
                "type": self.type,
                "cell_id": self.cell_id,
                "source": self.source,
                "execute": self.execute,
            }
        if self.type in {"delete_cell", "run_cell"}:
            return {"type": self.type, "cell_id": self.cell_id}
        if self.type == "move_cell":
            return {
                "type": self.type,
                "cell_id": self.cell_id,
                "new_position": self.new_position,
            }
        if self.type == "inspect_notebook":
            return {"type": self.type}
        if self.type == "restart_and_run_all":
            return {"type": self.type}
        if self.type == "validate":
            return {"type": self.type, "model_var": self.model_var}
        return {"type": "submit", "model_var": self.model_var}


@dataclass
class Observation:
    """Structured feedback returned after an environment action."""

    action: ActionType
    step: int
    budget_remaining: int
    code: str = ""
    stdout: str = ""
    stderr: str = ""
    hints: list[str] = field(default_factory=list)
    checklist_coverage: float = 0.0
    done: bool = False
    submitted: bool = False
    test_metric: float | None = None
    model_var: str | None = None
    cell_id: str | None = None
    notebook_status: dict[str, Any] = field(default_factory=dict)
    feedback_items: list[dict[str, Any]] = field(default_factory=list)
    validation_metric: float | None = None

    def to_feedback_message(self) -> str:
        parts = [f"[ACTION] {self.action}", f"[STEP] {self.step}"]

        if self.cell_id:
            parts.append(f"[CELL] {self.cell_id}")
        if self.stdout.strip():
            parts.append(f"[EXECUTION RESULT]\n{self.stdout.strip()}")
        if self.stderr.strip():
            parts.append(f"[CONTRACT FEEDBACK]\n{self.stderr.strip()}")
        if self.hints:
            hints_text = "\n".join(f"- {hint}" for hint in self.hints)
            parts.append(f"[CHECKLIST FEEDBACK]\n{hints_text}")
        if self.validation_metric is not None:
            parts.append(f"[VALIDATION] metric={self.validation_metric:.6f}")

        if self.notebook_status:
            status_text = "\n".join(
                f"{key}={value}" for key, value in self.notebook_status.items()
            )
            parts.append(f"[NOTEBOOK STATUS]\n{status_text}")
        parts.append(f"[BUDGET] {self.budget_remaining} code steps remaining.")

        if self.submitted:
            parts.append("[SUBMITTED] Final candidate accepted. Episode finished.")
        elif self.done:
            parts.append("[DONE] Step budget exhausted.")

        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "step": self.step,
            "budget_remaining": self.budget_remaining,
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "hints": self.hints,
            "checklist_coverage": self.checklist_coverage,
            "done": self.done,
            "submitted": self.submitted,
            "test_metric": self.test_metric,
            "model_var": self.model_var,
            "cell_id": self.cell_id,
            "notebook_status": self.notebook_status,
            "feedback_items": self.feedback_items,
            "validation_metric": self.validation_metric,
        }


StepResult = Observation


def coerce_action(action: Action | dict[str, Any] | str) -> Action:
    if isinstance(action, Action):
        return action
    if isinstance(action, dict):
        return Action.from_payload(action)
    if isinstance(action, str):
        return Action.from_llm_response(action)
    raise ActionParseError(f"Unsupported action object: {type(action).__name__}")


def _normalize_action_type(raw_type: Any) -> str:
    if raw_type is None:
        raise ActionParseError("Action is missing the 'type' field.")
    normalized = str(raw_type).strip().lower().replace("-", "_")
    aliases = {
        "write_code": "code",
        "run_code": "code",
        "python": "code",
        "add-code-cell": "add_cell",
        "add_code_cell": "add_cell",
        "add-markdown-cell": "add_cell",
        "run-cell": "run_cell",
        "run_all": "restart_and_run_all",
        "restart-run-all": "restart_and_run_all",
        "submit_model": "submit",
    }
    return aliases.get(normalized, normalized)


def _extract_json_block(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate

    return None


def _extract_code_block(text: str) -> str:
    for pattern in (
        r"```python\s*(.*?)\s*```",
        r"```py\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
    ):
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text.strip()

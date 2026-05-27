import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal


ActionType = Literal["code", "submit"]


class ActionParseError(ValueError):
    """Raised when an LLM response looks like an action but is invalid."""


ACTION_JSON_SCHEMA = """
Respond with exactly one JSON object.

Code action:
{"type": "code", "code": "print(train_df.shape)"}

Submit action:
{"type": "submit", "model_var": "best_model"}
""".strip()


@dataclass(frozen=True)
class Action:
    """One explicit agent action sent to the environment."""

    type: ActionType
    code: str = ""
    model_var: str = "model"

    def __post_init__(self) -> None:
        if self.type not in {"code", "submit"}:
            raise ActionParseError(f"Unsupported action type: {self.type!r}")
        if self.type == "code" and not isinstance(self.code, str):
            raise ActionParseError("Code action requires a string 'code' field.")
        if self.type == "submit" and not self.model_var:
            raise ActionParseError("Submit action requires a non-empty 'model_var'.")

    @classmethod
    def code_action(cls, code: str) -> "Action":
        return cls(type="code", code=code)

    @classmethod
    def submit_action(cls, model_var: str = "model") -> "Action":
        return cls(type="submit", model_var=model_var)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Action":
        raw_type = payload.get("type") or payload.get("action")
        action_type = _normalize_action_type(raw_type)

        if action_type == "code":
            code = payload.get("code")
            if code is None:
                raise ActionParseError("Code action is missing the 'code' field.")
            return cls.code_action(str(code))

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

    def to_dict(self) -> dict[str, str]:
        if self.type == "code":
            return {"type": "code", "code": self.code}
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

    def to_feedback_message(self) -> str:
        parts = [f"[ACTION] {self.action}", f"[STEP] {self.step}"]

        if self.stdout.strip():
            parts.append(f"[OUTPUT]\n{self.stdout.strip()}")
        if self.stderr.strip():
            parts.append(f"[ERROR]\n{self.stderr.strip()}")
        if self.hints:
            hints_text = "\n".join(f"- {hint}" for hint in self.hints)
            parts.append(f"[HINTS]\n{hints_text}")

        parts.append(f"[CHECKLIST] coverage={self.checklist_coverage:.2f}")
        parts.append(f"[BUDGET] {self.budget_remaining} code steps remaining.")

        if self.submitted:
            parts.append(f"[SUBMITTED] test_metric={self.test_metric}")
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

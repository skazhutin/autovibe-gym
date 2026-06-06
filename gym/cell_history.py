from dataclasses import dataclass, field
from typing import Any

from .protocol import ActionType, Observation


@dataclass(frozen=True)
class Cell:
    """One notebook-like record of an environment action."""

    execution_count: int
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_count": self.execution_count,
            "type": self.action,
            "step": self.step,
            "budget_remaining": self.budget_remaining,
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "hints": list(self.hints),
            "checklist_coverage": self.checklist_coverage,
            "done": self.done,
            "submitted": self.submitted,
            "model_var": self.model_var,
        }

    def to_private_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data["test_metric"] = self.test_metric
        return data


@dataclass
class CellHistory:
    """Notebook-style execution history kept alongside the runtime workspace."""

    cells: list[Cell] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cells)

    def reset(self) -> None:
        self.cells = []

    def append_observation(self, observation: Observation) -> Cell:
        cell = Cell(
            execution_count=len(self.cells) + 1,
            action=observation.action,
            step=observation.step,
            budget_remaining=observation.budget_remaining,
            code=observation.code,
            stdout=observation.stdout,
            stderr=observation.stderr,
            hints=list(observation.hints),
            checklist_coverage=observation.checklist_coverage,
            done=observation.done,
            submitted=observation.submitted,
            test_metric=observation.test_metric,
            model_var=observation.model_var,
        )
        self.cells.append(cell)
        return cell

    def code_cells(self) -> list[Cell]:
        return [cell for cell in self.cells if cell.action == "code"]

    def last(self) -> Cell | None:
        if not self.cells:
            return None
        return self.cells[-1]

    def to_dict(self) -> list[dict[str, Any]]:
        return [cell.to_dict() for cell in self.cells]

    def to_feedback_context(
        self,
        *,
        max_cells: int = 4,
        max_code_chars: int = 900,
        max_output_chars: int = 500,
    ) -> str:
        if not self.cells:
            return ""

        recent = self.cells[-max_cells:]
        lines = ["[NOTEBOOK] Recent executed cells in this session:"]
        for cell in recent:
            status = self._status_label(cell)
            lines.append(
                f"In [{cell.execution_count}] {cell.action} "
                f"step={cell.step} status={status}"
            )
            if cell.code.strip():
                lines.append("code:")
                lines.append(_indent(_clip(cell.code.strip(), max_code_chars)))
            if cell.stdout.strip():
                lines.append("output:")
                lines.append(_indent(_clip(cell.stdout.strip(), max_output_chars)))
            if cell.stderr.strip():
                lines.append("error:")
                lines.append(_indent(_clip(cell.stderr.strip(), max_output_chars)))
            if cell.submitted:
                lines.append(f"submitted model={cell.model_var}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = ["# AutoVibe Gym Cell History", ""]
        if not self.cells:
            lines.append("_No cells executed._")
            return "\n".join(lines)

        for cell in self.cells:
            status = self._status_label(cell)
            lines.extend([
                f"## In [{cell.execution_count}] {cell.action} step={cell.step}",
                "",
                f"- Status: {status}",
                f"- Budget remaining: {cell.budget_remaining}",
                f"- Checklist coverage: {cell.checklist_coverage:.2f}",
            ])
            if cell.submitted:
                lines.append(f"- Submitted model: {cell.model_var}")
            if cell.code.strip():
                lines.extend(["", "```python", cell.code.strip(), "```"])
            if cell.stdout.strip():
                lines.extend(["", "**stdout**", "", "```text", cell.stdout.strip(), "```"])
            if cell.stderr.strip():
                lines.extend(["", "**stderr**", "", "```text", cell.stderr.strip(), "```"])
            if cell.hints:
                lines.extend(["", "**pending hints**"])
                lines.extend(f"- {hint}" for hint in cell.hints)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _status_label(cell: Cell) -> str:
        if cell.submitted:
            return "submitted"
        if cell.stderr.strip():
            return "error"
        if cell.done:
            return "done"
        return "ok"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... [truncated]"


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())

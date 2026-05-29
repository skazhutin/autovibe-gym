from __future__ import annotations

from dataclasses import dataclass

from .feedback import GENERIC_CHECKLIST_HINTS, MANDATORY_CHECKS, NotebookChecklist


@dataclass
class CheckItem:
    key: str
    passed: bool = False
    hint: str = ""


class Checklist:
    """
    Backward-compatible wrapper around the generic hidden notebook checklist.

    Hints are deliberately generic and selective. Dataset-specific facts and
    implementation-prescriptive suggestions are not emitted to the agent.
    """

    def __init__(self, target_col: str):
        self.target_col = target_col
        self._delegate = NotebookChecklist(target_col=target_col)
        self.items: dict[str, CheckItem] = {
            key: CheckItem(key=key, hint=GENERIC_CHECKLIST_HINTS[key])
            for key in MANDATORY_CHECKS
        }

    def evaluate(
        self,
        code: str,
        stdout: str,
        namespace: dict,
        history: list,
    ) -> list[str]:
        items = self._delegate.record_execution(
            source=code,
            stdout=stdout,
            cell_id=None,
            step=len(history) + 1,
            execution_success=True,
        )
        self._sync_items()
        return [item.message for item in items]

    def record_structural(self, key: str, *, reason: str, step: int = 0) -> None:
        self._delegate.record_structural(key, reason=reason, step=step)
        self._sync_items()

    def coverage(self) -> float:
        return self._delegate.coverage()

    def _sync_items(self) -> None:
        for key, item in self.items.items():
            item.passed = key in self._delegate.covered

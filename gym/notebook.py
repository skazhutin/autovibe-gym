from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat
from nbformat import NotebookNode


@dataclass
class NotebookDocument:
    path: Path
    notebook: NotebookNode

    @classmethod
    def create(cls, path: str | Path) -> "NotebookDocument":
        nb = nbformat.v4.new_notebook()
        nb.metadata["autovibe"] = {
            "revision": 0,
            "next_cell_index": 1,
            "protocol_version": "jupyter-v1",
        }
        doc = cls(path=Path(path), notebook=nb)
        doc.save()
        return doc

    @classmethod
    def load(cls, path: str | Path) -> "NotebookDocument":
        nb_path = Path(path)
        nb = nbformat.read(nb_path, as_version=4)
        nb.metadata.setdefault("autovibe", {})
        nb.metadata["autovibe"].setdefault("revision", 0)
        nb.metadata["autovibe"].setdefault("next_cell_index", len(nb.cells) + 1)
        return cls(path=nb_path, notebook=nb)

    @property
    def revision(self) -> int:
        return int(self.notebook.metadata.get("autovibe", {}).get("revision", 0))

    def add_code_cell(
        self,
        source: str,
        *,
        position: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        cell_id = self._next_cell_id()
        cell = nbformat.v4.new_code_cell(source=source, metadata=metadata or {})
        self._initialize_cell(cell, cell_id)
        self._insert_cell(cell, position)
        self._bump_revision()
        self.save()
        return cell_id

    def add_markdown_cell(
        self,
        source: str,
        *,
        position: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        cell_id = self._next_cell_id()
        cell = nbformat.v4.new_markdown_cell(source=source, metadata=metadata or {})
        self._initialize_cell(cell, cell_id)
        self._insert_cell(cell, position)
        self._bump_revision()
        self.save()
        return cell_id

    def update_cell(self, cell_id: str, source: str) -> dict[str, Any]:
        index, cell = self._find_cell(cell_id)
        before = self._cell_snapshot(cell)
        cell.source = source
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        self._bump_revision(cell)
        self.save()
        return {"index": index, "before": before, "after": self._cell_snapshot(cell)}

    def delete_cell(self, cell_id: str) -> dict[str, Any]:
        index, cell = self._find_cell(cell_id)
        before = self._cell_snapshot(cell)
        del self.notebook.cells[index]
        self._bump_revision()
        self.save()
        return {"index": index, "before": before}

    def move_cell(self, cell_id: str, new_position: int) -> dict[str, Any]:
        old_index, cell = self._find_cell(cell_id)
        if new_position < 0 or new_position >= len(self.notebook.cells):
            raise IndexError(f"new_position out of range: {new_position}")
        del self.notebook.cells[old_index]
        self.notebook.cells.insert(new_position, cell)
        self._bump_revision(cell)
        self.save()
        return {
            "cell_id": cell_id,
            "old_position": old_index,
            "new_position": new_position,
        }

    def get_cell(self, cell_id: str) -> NotebookNode:
        _, cell = self._find_cell(cell_id)
        return cell

    def list_cells(self) -> list[dict[str, Any]]:
        return [self._compact_cell(cell, index) for index, cell in enumerate(self.notebook.cells)]

    def clear_outputs(self) -> None:
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        self.save()

    def set_cell_outputs(
        self,
        cell_id: str,
        *,
        outputs: list[dict[str, Any]],
        execution_count: int | None,
    ) -> None:
        _, cell = self._find_cell(cell_id)
        if cell.cell_type != "code":
            raise TypeError(f"Cell {cell_id!r} is not a code cell.")
        cell.outputs = [nbformat.from_dict(copy.deepcopy(output)) for output in outputs]
        cell.execution_count = execution_count
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(self.notebook, self.path)

    def validate(self) -> None:
        nbformat.validate(self.notebook)

    def export_python(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks: list[str] = []
        for cell in self.notebook.cells:
            if cell.cell_type != "code":
                continue
            cell_id = str(cell.get("id", "unknown"))
            chunks.append(f"# %% [{cell_id}]\n{cell.source.rstrip()}\n")
        output_path.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")
        return output_path

    def _insert_cell(self, cell: NotebookNode, position: int | None) -> None:
        if position is None:
            self.notebook.cells.append(cell)
            return
        if position < 0 or position > len(self.notebook.cells):
            raise IndexError(f"position out of range: {position}")
        self.notebook.cells.insert(position, cell)

    def _initialize_cell(self, cell: NotebookNode, cell_id: str) -> None:
        cell["id"] = cell_id
        cell.metadata.setdefault("autovibe", {})
        cell.metadata["autovibe"].setdefault("revision", 1)

    def _next_cell_id(self) -> str:
        meta = self.notebook.metadata.setdefault("autovibe", {})
        next_index = int(meta.get("next_cell_index", 1))
        meta["next_cell_index"] = next_index + 1
        return f"cell_{next_index:02d}"

    def _bump_revision(self, cell: NotebookNode | None = None) -> None:
        meta = self.notebook.metadata.setdefault("autovibe", {})
        meta["revision"] = int(meta.get("revision", 0)) + 1
        if cell is not None:
            cell.metadata.setdefault("autovibe", {})
            cell_meta = cell.metadata["autovibe"]
            cell_meta["revision"] = int(cell_meta.get("revision", 0)) + 1

    def _find_cell(self, cell_id: str) -> tuple[int, NotebookNode]:
        for index, cell in enumerate(self.notebook.cells):
            if str(cell.get("id")) == cell_id:
                return index, cell
        raise KeyError(f"Unknown cell_id: {cell_id}")

    def _cell_snapshot(self, cell: NotebookNode) -> dict[str, Any]:
        return {
            "cell_id": str(cell.get("id")),
            "cell_type": str(cell.cell_type),
            "source": str(cell.source),
            "execution_count": cell.get("execution_count"),
            "outputs": copy.deepcopy(cell.get("outputs", [])),
            "metadata": copy.deepcopy(cell.get("metadata", {})),
        }

    def _compact_cell(self, cell: NotebookNode, index: int) -> dict[str, Any]:
        outputs = cell.get("outputs", []) if cell.cell_type == "code" else []
        status = "not_run"
        if any(output.get("output_type") == "error" for output in outputs):
            status = "error"
        elif outputs or cell.get("execution_count") is not None:
            status = "ok"

        source = str(cell.source)
        preview = source.replace("\n", "\\n")
        if len(preview) > 120:
            preview = preview[:117].rstrip() + "..."

        return {
            "index": index,
            "cell_id": str(cell.get("id")),
            "cell_type": str(cell.cell_type),
            "source_preview": preview,
            "execution_count": cell.get("execution_count"),
            "status": status,
            "revision": int(cell.metadata.get("autovibe", {}).get("revision", 0)),
        }

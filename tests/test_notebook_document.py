import nbformat
import pytest

from gym.notebook import NotebookDocument


def test_notebook_document_add_update_delete_move_and_export(tmp_path):
    path = tmp_path / "solution.ipynb"
    doc = NotebookDocument.create(path)

    code_id = doc.add_code_cell("x = 1")
    markdown_id = doc.add_markdown_cell("## Audit")

    assert path.exists()
    assert doc.list_cells()[0]["cell_id"] == code_id
    assert doc.list_cells()[1]["cell_type"] == "markdown"

    before_revision = doc.notebook.cells[0].metadata["autovibe"]["revision"]
    edit = doc.update_cell(code_id, "x = 2\nprint(x)")
    assert edit["before"]["source"] == "x = 1"
    assert doc.notebook.cells[0].metadata["autovibe"]["revision"] == before_revision + 1

    doc.move_cell(markdown_id, 0)
    assert doc.list_cells()[0]["cell_id"] == markdown_id

    doc.set_cell_outputs(
        code_id,
        outputs=[{"output_type": "stream", "name": "stdout", "text": "2\n"}],
        execution_count=1,
    )
    assert doc.get_cell(code_id).execution_count == 1
    assert doc.list_cells()[1]["status"] == "ok"

    deleted = doc.delete_cell(markdown_id)
    assert deleted["before"]["cell_type"] == "markdown"
    assert len(doc.list_cells()) == 1

    exported = doc.export_python(tmp_path / "solution.py")
    assert "print(x)" in exported.read_text(encoding="utf-8")

    loaded = nbformat.read(path, as_version=4)
    nbformat.validate(loaded)


def test_notebook_document_edge_cases_and_round_trip(tmp_path):
    path = tmp_path / "edge.ipynb"
    doc = NotebookDocument.create(path)

    empty_id = doc.add_code_cell("")
    multiline_id = doc.add_code_cell("a = 1\nb = 2\nprint(a + b)", position=0)
    long_id = doc.add_markdown_cell("x" * 200)

    assert doc.list_cells()[0]["cell_id"] == multiline_id
    assert doc.list_cells()[-1]["source_preview"].endswith("...")

    with pytest.raises(IndexError):
        doc.add_code_cell("bad", position=99)
    with pytest.raises(KeyError, match="Unknown cell_id"):
        doc.get_cell("missing")
    with pytest.raises(IndexError):
        doc.move_cell(empty_id, -1)

    doc.set_cell_outputs(
        multiline_id,
        outputs=[{"output_type": "stream", "name": "stdout", "text": "3\n"}],
        execution_count=7,
    )
    assert doc.get_cell(multiline_id).execution_count == 7
    edit = doc.update_cell(multiline_id, "print('fresh')")
    assert edit["before"]["outputs"]
    assert doc.get_cell(multiline_id).outputs == []
    assert doc.get_cell(multiline_id).execution_count is None

    with pytest.raises(TypeError):
        doc.set_cell_outputs(
            long_id,
            outputs=[{"output_type": "stream", "name": "stdout", "text": "x"}],
            execution_count=1,
        )

    doc.move_cell(long_id, 0)
    assert doc.list_cells()[0]["cell_id"] == long_id
    doc.move_cell(long_id, len(doc.list_cells()) - 1)
    assert doc.list_cells()[-1]["cell_id"] == long_id

    loaded = NotebookDocument.load(path)
    loaded.validate()
    exported = loaded.export_python(tmp_path / "edge.py").read_text(encoding="utf-8")
    assert exported.index("print('fresh')") < exported.rindex("# %%")


def test_load_notebook_without_autovibe_metadata(tmp_path):
    path = tmp_path / "plain.ipynb"
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("print('plain')"))
    nbformat.write(nb, path)

    doc = NotebookDocument.load(path)

    assert doc.revision == 0
    new_id = doc.add_code_cell("print('next')")
    assert new_id == "cell_02"

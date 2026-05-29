import nbformat

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

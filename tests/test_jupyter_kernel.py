import pandas as pd

from gym.jupyter_kernel import JupyterKernelSession


def _write_bootstrap_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    val = pd.DataFrame({"x": [3], "target": [1]})
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    return train_path, val_path


def test_kernel_bootstrap_persistence_stdout_and_shutdown(tmp_path):
    train_path, val_path = _write_bootstrap_data(tmp_path)
    session = JupyterKernelSession(tmp_path, timeout=30)
    try:
        session.start()
        session.inject_bootstrap_context(
            train_csv=train_path,
            val_csv=val_path,
            target_col="target",
        )

        result = session.execute_cell(
            "value = 41\nprint(train_df.shape)\nprint('test_df' in globals())"
        )
        assert result.success
        assert "(2, 2)" in result.stdout
        assert "False" in result.stdout

        persisted = session.execute_cell("print(value + 1)")
        assert persisted.success
        assert "42" in persisted.stdout
    finally:
        session.shutdown()


def test_kernel_captures_errors_display_outputs_and_plots(tmp_path):
    session = JupyterKernelSession(tmp_path, timeout=30)
    try:
        session.start()
        display_result = session.execute_cell(
            "from IPython.display import display\n"
            "display({'a': 1})\n"
            "import matplotlib.pyplot as plt\n"
            "plt.plot([1, 2, 3])\n"
            "plt.show()"
        )
        assert display_result.success
        assert any(
            output["output_type"] == "display_data"
            for output in display_result.outputs
        )

        error_result = session.execute_cell("raise ValueError('boom')")
        assert not error_result.success
        assert error_result.error_name == "ValueError"
        assert "boom" in error_result.error_value
        assert error_result.traceback
    finally:
        session.shutdown()

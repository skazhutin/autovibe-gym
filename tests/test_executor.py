"""Tests for CodeExecutor — no LLM or server required."""
import types

import pytest
from gym.executor import CodeExecutor


@pytest.fixture
def executor():
    return CodeExecutor(timeout=10, backend="subprocess")


def test_basic_execution(executor):
    stdout, stderr, ns = executor.run("x = 42\nprint(x)")
    assert "42" in stdout
    assert stderr == ""


def test_namespace_in_out(executor):
    """Variables passed in are accessible; new variables are returned."""
    ns_in = {"a": 10, "b": 20}
    stdout, stderr, ns_out = executor.run("c = a + b\nprint(c)", ns_in)
    assert ns_out["c"] == 30
    assert "30" in stdout


def test_namespace_persists_across_calls(executor):
    """Simulates multi-step: ns from step 1 flows into step 2."""
    _, _, ns = executor.run("import numpy as np\narr = np.array([1, 2, 3])")
    assert "arr" in ns
    stdout, stderr, ns2 = executor.run("print(arr.sum())", ns)
    assert "6" in stdout


def test_syntax_error_captured(executor):
    stdout, stderr, ns = executor.run("def broken(:\n    pass")
    assert stderr.strip() != ""


def test_runtime_error_captured(executor):
    stdout, stderr, ns = executor.run("1 / 0")
    assert "ZeroDivisionError" in stderr


def test_timeout(tmp_path):
    fast_executor = CodeExecutor(timeout=2, backend="subprocess")
    stdout, stderr, ns = fast_executor.run("import time\ntime.sleep(10)")
    assert "timed out" in stderr.lower()


def test_import_pandas(executor):
    stdout, stderr, ns = executor.run(
        "import pandas as pd\ndf = pd.DataFrame({'a': [1,2,3]})\nprint(df.shape)"
    )
    assert "(3, 1)" in stdout
    assert stderr == ""


def test_cwd_is_tmpdir_not_project_root(tmp_path):
    """Subprocess cwd must be an isolated tmpdir — not the caller's directory.

    This is the core privacy guarantee: agent code cannot read test.csv by
    relative path even if test.csv sits alongside the training data.
    """
    # Write a sentinel file in a test-controlled directory
    sentinel = tmp_path / "test.csv"
    sentinel.write_text("secret_label\n1\n0\n")

    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)           # pretend project cwd contains test.csv
    try:
        executor = CodeExecutor(timeout=10)
        stdout, stderr, ns = executor.run(
            "import pandas as pd\ndf = pd.read_csv('test.csv')\nprint('LEAKED')"
        )
    finally:
        os.chdir(original_cwd)

    assert "LEAKED" not in stdout, "test.csv was readable — cwd isolation failed"
    assert stderr.strip() != "", "Expected FileNotFoundError in stderr"


def test_sklearn_model_survives_pickle(executor):
    """Trained sklearn model must survive namespace serialisation."""
    code = """
from sklearn.linear_model import LogisticRegression
import numpy as np
X = np.array([[1,2],[3,4],[5,6],[7,8]])
y = np.array([0, 0, 1, 1])
model = LogisticRegression()
model.fit(X, y)
print("fitted")
"""
    stdout, stderr, ns = executor.run(code)
    assert "fitted" in stdout
    assert "model" in ns
    assert hasattr(ns["model"], "predict")


def test_subprocess_backend_does_not_inherit_secret_env(monkeypatch, executor):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-token")

    stdout, stderr, ns = executor.run(
        "import os\nprint(os.getenv('GEMINI_API_KEY'))"
    )

    assert "None" in stdout
    assert "secret-token" not in stdout
    assert stderr == ""


def test_subprocess_backend_blocks_network(executor):
    stdout, stderr, ns = executor.run(
        "import socket\nsocket.create_connection(('example.com', 80), timeout=1)"
    )

    assert "Network access is disabled" in stderr


def test_subprocess_backend_blocks_external_file_reads(tmp_path, executor):
    secret = tmp_path / "secret.txt"
    secret.write_text("hidden", encoding="utf-8")

    stdout, stderr, ns = executor.run(f"open({str(secret)!r}).read()")

    assert "Sandbox read outside allowed roots is blocked" in stderr


def test_docker_backend_builds_locked_down_command(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return types.SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr("gym.executor.subprocess.run", fake_run)
    executor = CodeExecutor(
        timeout=10,
        backend="docker",
        docker_image="autovibe-test-sandbox",
    )

    stdout, stderr, ns = executor.run("x = 1")

    command = captured["command"]
    joined = " ".join(command)
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--memory 2048m" in joined
    assert "--pids-limit 128" in joined
    assert "--entrypoint python" in joined
    assert "autovibe-test-sandbox" in command
    assert "GEMINI_API_KEY" not in joined


def test_docker_backend_reports_missing_cli(monkeypatch):
    def fake_run(command, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("gym.executor.subprocess.run", fake_run)
    executor = CodeExecutor(timeout=10, backend="docker")

    stdout, stderr, ns = executor.run("print(1)")

    assert stdout == ""
    assert "Docker backend is enabled" in stderr

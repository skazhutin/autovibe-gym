import types

import pandas as pd
import pytest

from gym.jupyter_kernel import (
    ContainerJupyterKernelBackend,
    ContainerJupyterKernelSession,
    JupyterKernelSession,
    _find_free_ports,
)


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


# ---------------------------------------------------------------------------
# ContainerJupyterKernelSession — mock-based unit tests (no Docker required)
# ---------------------------------------------------------------------------

def _make_fake_subprocess(container_id="abc123", returncode=0):
    """Return a fake subprocess.run that records calls and returns a fixed result."""
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(
            stdout=container_id + "\n",
            stderr="",
            returncode=returncode,
        )

    return fake_run, calls


def test_container_session_docker_command_includes_security_flags(monkeypatch, tmp_path):
    fake_run, calls = _make_fake_subprocess()

    # Patch subprocess.run and BlockingKernelClient so no real Docker/ZMQ is used
    monkeypatch.setattr("gym.jupyter_kernel.subprocess.run", fake_run)

    fake_client = types.SimpleNamespace(
        load_connection_info=lambda info: None,
        start_channels=lambda: None,
        wait_for_ready=lambda timeout=30: None,
        stop_channels=lambda: None,
    )
    monkeypatch.setattr(
        "gym.jupyter_kernel.BlockingKernelClient",
        lambda: fake_client,
    )

    session = ContainerJupyterKernelSession(
        tmp_path,
        docker_image="autovibe-test-sandbox",
        memory_mb=1024,
        cpus="0.5",
        pids_limit=256,
        network_name="autovibe-kernels",
    )
    session.start()

    assert calls, "subprocess.run was never called"
    docker_cmd = calls[0]
    joined = " ".join(str(t) for t in docker_cmd)

    assert docker_cmd[:3] == ["docker", "run", "-d"]
    assert "--read-only" in docker_cmd
    assert "--cap-drop" in docker_cmd and "ALL" in docker_cmd
    assert "--security-opt" in docker_cmd and "no-new-privileges" in docker_cmd
    assert "--memory" in docker_cmd and "1024m" in joined
    assert "--pids-limit" in docker_cmd and "256" in joined
    assert "--network" in docker_cmd and "autovibe-kernels" in joined
    assert "--cpus" in docker_cmd and "0.5" in joined
    # 5 ZMQ ports must be published
    assert joined.count("-p ") >= 5 or joined.count(" -p ") >= 5 or docker_cmd.count("-p") >= 5
    assert "autovibe-test-sandbox" in docker_cmd
    assert "ipykernel_launcher" in joined


def test_container_session_shutdown_stops_container(monkeypatch, tmp_path):
    fake_run, calls = _make_fake_subprocess(container_id="mycontainer99")

    monkeypatch.setattr("gym.jupyter_kernel.subprocess.run", fake_run)

    fake_client = types.SimpleNamespace(
        load_connection_info=lambda info: None,
        start_channels=lambda: None,
        wait_for_ready=lambda timeout=30: None,
        stop_channels=lambda: None,
    )
    monkeypatch.setattr("gym.jupyter_kernel.BlockingKernelClient", lambda: fake_client)

    session = ContainerJupyterKernelSession(tmp_path, docker_image="autovibe-test-sandbox")
    session.start()

    calls.clear()
    session.shutdown()

    assert any(
        "stop" in " ".join(str(t) for t in cmd) and "mycontainer99" in " ".join(str(t) for t in cmd)
        for cmd in calls
    ), f"Expected 'docker stop mycontainer99' in calls, got: {calls}"

    assert session.client is None
    assert session._container_id is None


def test_container_session_raises_if_docker_unavailable(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr("gym.jupyter_kernel.subprocess.run", fake_run)

    session = ContainerJupyterKernelSession(tmp_path, docker_image="autovibe-test-sandbox")
    with pytest.raises((FileNotFoundError, RuntimeError)):
        session.start()


def test_container_backend_ensure_network_called_on_init(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("gym.jupyter_kernel.subprocess.run", fake_run)

    ContainerJupyterKernelBackend(network_name="test-net")

    assert any(
        "network" in " ".join(str(t) for t in cmd) and "test-net" in " ".join(str(t) for t in cmd)
        for cmd in calls
    ), f"Expected docker network create in calls, got: {calls}"


def test_container_backend_raises_if_docker_missing(monkeypatch):
    def fake_run(command, **kwargs):
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr("gym.jupyter_kernel.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="docker"):
        ContainerJupyterKernelBackend()


def test_find_free_ports_returns_distinct_ports():
    ports = _find_free_ports(5)
    assert len(ports) == 5
    assert len(set(ports)) == 5
    for port in ports:
        assert 1024 <= port <= 65535


# ---------------------------------------------------------------------------
# Integration test — requires a running Docker daemon and built sandbox image
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    import subprocess as sp
    try:
        result = sp.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _docker_available(), reason="Docker not available")
def test_container_session_integration(tmp_path):
    """End-to-end: start kernel in container, inject bootstrap, execute cell, shutdown."""
    train_path, val_path = _write_bootstrap_data(tmp_path)

    session = ContainerJupyterKernelSession(
        tmp_path,
        timeout=60,
        docker_image="autovibe-gym-sandbox:latest",
    )
    try:
        session.start()
        session.inject_bootstrap_context(
            train_csv=train_path,
            val_csv=val_path,
            target_col="target",
        )

        result = session.execute_cell("print(train_df.shape)")
        assert result.success, f"Cell failed: {result.error_name}: {result.error_value}"
        assert "(2, 2)" in result.stdout

        # Verify internet is blocked inside the container
        net_result = session.execute_cell(
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('8.8.8.8', 53), timeout=2)\n"
            "    print('CONNECTED')\n"
            "except OSError:\n"
            "    print('BLOCKED')\n"
        )
        assert "BLOCKED" in net_result.stdout, (
            "Network should be blocked inside container kernel"
        )
    finally:
        session.shutdown()

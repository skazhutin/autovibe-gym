from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from jupyter_client import BlockingKernelClient, KernelManager
from nbformat.v4 import output_from_msg


@dataclass
class CellExecutionResult:
    execution_count: int | None = None
    outputs: list[dict[str, Any]] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    error_name: str | None = None
    error_value: str | None = None
    traceback: list[str] = field(default_factory=list)
    display_outputs: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_count": self.execution_count,
            "outputs": self.outputs,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error_name": self.error_name,
            "error_value": self.error_value,
            "traceback": self.traceback,
            "display_outputs": self.display_outputs,
            "success": self.success,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def compact_text(self, *, max_chars: int = 2000) -> str:
        chunks: list[str] = []
        if self.stdout.strip():
            chunks.append(self.stdout.strip())
        if self.stderr.strip():
            chunks.append(f"[stderr]\n{self.stderr.strip()}")
        for output in self.outputs:
            if output.get("output_type") in {"execute_result", "display_data"}:
                text = output.get("data", {}).get("text/plain")
                if isinstance(text, list):
                    text = "".join(text)
                if text:
                    chunks.append(str(text).strip())
            elif output.get("output_type") == "error":
                chunks.append(
                    f"{output.get('ename')}: {output.get('evalue')}".strip()
                )
        text = "\n\n".join(chunk for chunk in chunks if chunk)
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "... [truncated]"


class KernelExecutionBackend(Protocol):
    def create_session(self, workspace_dir: str | Path) -> "JupyterKernelSession":
        ...


class _KernelClientMixin:
    """Shared ZMQ client methods used by both local and container kernel sessions."""

    client: Any
    workspace_dir: Path
    timeout: int

    def execute_cell(
        self,
        source: str,
        *,
        timeout: int | None = None,
        store_history: bool = True,
    ) -> CellExecutionResult:
        if self.client is None:
            raise RuntimeError("Jupyter kernel is not started.")

        effective_timeout = timeout or self.timeout
        started = time.time()
        msg_id = self.client.execute(
            source,
            store_history=store_history,
            allow_stdin=False,
        )
        result = CellExecutionResult()

        while True:
            if time.time() - started > effective_timeout:
                self._interrupt_kernel()
                result.success = False
                result.error_name = "TimeoutError"
                result.error_value = f"Cell execution exceeded {effective_timeout}s"
                result.stderr = result.error_value
                result.outputs.append(
                    {
                        "output_type": "error",
                        "ename": result.error_name,
                        "evalue": result.error_value,
                        "traceback": [result.error_value],
                    }
                )
                break

            try:
                msg = self.client.get_iopub_msg(timeout=1)
            except Exception:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "status" and content.get("execution_state") == "idle":
                break
            if msg_type == "execute_input":
                result.execution_count = content.get("execution_count")
                continue
            if msg_type not in {
                "stream",
                "execute_result",
                "display_data",
                "error",
            }:
                continue

            output = _output_from_message(msg)
            if output is not None:
                result.outputs.append(output)

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stderr":
                    result.stderr += text
                else:
                    result.stdout += text
            elif msg_type in {"execute_result", "display_data"}:
                if msg_type == "execute_result":
                    result.execution_count = content.get(
                        "execution_count",
                        result.execution_count,
                    )
                result.display_outputs.append(output or {})
            elif msg_type == "error":
                result.success = False
                result.error_name = content.get("ename")
                result.error_value = content.get("evalue")
                result.traceback = list(content.get("traceback") or [])
                result.stderr += "\n".join(result.traceback)

        result.elapsed_seconds = round(time.time() - started, 3)
        return result

    def inject_bootstrap_context(
        self,
        *,
        train_csv: str | Path,
        val_csv: str | Path,
        target_col: str,
    ) -> CellExecutionResult:
        source = f"""
import pandas as pd
import numpy as np
from pathlib import Path

_AUTOVIBE_WORKSPACE = Path({str(self.workspace_dir)!r})
train_df = pd.read_csv(Path({str(train_csv)!r}))
val_df = pd.read_csv(Path({str(val_csv)!r}))
target_col = {target_col!r}
""".strip()
        result = self.execute_cell(source, store_history=False)
        if not result.success:
            raise RuntimeError(
                "Failed to inject Jupyter bootstrap context: "
                f"{result.error_name}: {result.error_value}"
            )
        return result

    def dump_variable_to_file(self, variable_name: str, output_path: str | Path) -> None:
        source = f"""
import pickle as _autovibe_pickle
from pathlib import Path as _AutovibePath

if {variable_name!r} not in globals():
    raise NameError("Variable not found: {variable_name}")

with open(_AutovibePath({str(output_path)!r}), "wb") as _autovibe_file:
    _autovibe_pickle.dump(globals()[{variable_name!r}], _autovibe_file)
""".strip()
        result = self.execute_cell(source, store_history=False)
        if not result.success:
            raise RuntimeError(
                f"{result.error_name or 'RuntimeError'}: "
                f"{result.error_value or result.stderr}"
            )

    def _interrupt_kernel(self) -> None:
        pass


class LocalJupyterKernelBackend:
    def create_session(self, workspace_dir: str | Path) -> "JupyterKernelSession":
        return JupyterKernelSession(workspace_dir=workspace_dir)


class JupyterKernelSession(_KernelClientMixin):
    def __init__(
        self,
        workspace_dir: str | Path,
        *,
        timeout: int = 60,
        env: dict[str, str] | None = None,
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.timeout = timeout
        self.env = env or _minimal_kernel_env()
        self.kernel_id = str(uuid.uuid4())
        self.km: KernelManager | None = None
        self.client = None

    def start(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        km = KernelManager()
        km.kernel_cmd = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        km.start_kernel(cwd=str(self.workspace_dir), env=self.env)
        client = km.client()
        client.start_channels()
        client.wait_for_ready(timeout=self.timeout)
        self.km = km
        self.client = client

    def shutdown(self) -> None:
        if self.client is not None:
            try:
                self.client.stop_channels()
            except Exception:
                pass
        if self.km is not None:
            try:
                self.km.shutdown_kernel(now=True)
            except Exception:
                pass
        self.km = None
        self.client = None

    def restart(self) -> None:
        self.shutdown()
        self.kernel_id = str(uuid.uuid4())
        self.start()

    def _interrupt_kernel(self) -> None:
        if self.km is None:
            return
        try:
            self.km.interrupt_kernel()
        except Exception:
            pass


class ContainerJupyterKernelSession(_KernelClientMixin):
    """Jupyter kernel session running inside an isolated Docker container.

    The container has no internet access (internal Docker network), a read-only
    rootfs, capped memory/CPU/pids, and can only write to /workspace and /tmp.
    The host communicates with the kernel over ZMQ ports published from the
    container to 127.0.0.1.
    """

    _CONN_FILE = ".kernel.json"

    def __init__(
        self,
        workspace_dir: str | Path,
        *,
        timeout: int = 60,
        docker_image: str = "autovibe-gym-sandbox:latest",
        memory_mb: int = 2048,
        cpus: str = "1",
        pids_limit: int = 512,
        network_name: str = "autovibe-kernels",
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.timeout = timeout
        self.docker_image = docker_image
        self.memory_mb = memory_mb
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.network_name = network_name
        self.kernel_id = str(uuid.uuid4())
        self._container_id: str | None = None
        self.client: BlockingKernelClient | None = None

    def start(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        ports = _find_free_ports(5)
        shell_port, iopub_port, stdin_port, control_port, hb_port = ports
        key = str(uuid.uuid4())

        conn_info = {
            "shell_port": shell_port,
            "iopub_port": iopub_port,
            "stdin_port": stdin_port,
            "control_port": control_port,
            "hb_port": hb_port,
            "ip": "0.0.0.0",
            "key": key,
            "transport": "tcp",
            "signature_scheme": "hmac-sha256",
            "kernel_name": "",
        }
        conn_file = self.workspace_dir / self._CONN_FILE
        conn_file.write_text(json.dumps(conn_info), encoding="utf-8")

        container_name = f"autovibe-kernel-{self.kernel_id}"
        command = self._build_docker_command(container_name, ports)

        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to start kernel container: {proc.stderr.strip()}"
            )
        self._container_id = proc.stdout.strip()

        kc = BlockingKernelClient()
        kc.load_connection_info({**conn_info, "ip": "127.0.0.1"})
        kc.start_channels()
        kc.wait_for_ready(timeout=self.timeout)
        self.client = kc

    def shutdown(self) -> None:
        if self.client is not None:
            try:
                self.client.stop_channels()
            except Exception:
                pass
            self.client = None
        if self._container_id is not None:
            try:
                subprocess.run(
                    [_docker_binary(), "stop", self._container_id],
                    capture_output=True,
                    timeout=15,
                )
            except Exception:
                pass
            self._container_id = None

    def restart(self) -> None:
        self.shutdown()
        self.kernel_id = str(uuid.uuid4())
        self.start()

    def _interrupt_kernel(self) -> None:
        if self._container_id is None:
            return
        try:
            subprocess.run(
                [_docker_binary(), "kill", "--signal", "SIGINT", self._container_id],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

    def _build_docker_command(self, container_name: str, ports: list[int]) -> list[str]:
        command = [
            _docker_binary(),
            "run",
            "-d",
            "--rm",
            "--name", container_name,
            "--workdir", "/workspace",
            "--mount", f"type=bind,source={self.workspace_dir},target=/workspace",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", f"{self.memory_mb}m",
            "--cpus", str(self.cpus),
            "--pids-limit", str(self.pids_limit),
            "--network", self.network_name,
        ]
        for port in ports:
            command.extend(["-p", f"{port}:{port}"])
        command.extend([
            "--env", "PYTHONIOENCODING=utf-8",
            "--env", "PYTHONUTF8=1",
            "--env", "MPLBACKEND=Agg",
            "--env", "AUTOVIBE_JUPYTER_KERNEL=1",
            self.docker_image,
            "python", "-m", "ipykernel_launcher", "-f", f"/workspace/{self._CONN_FILE}",
        ])
        return command


class ContainerJupyterKernelBackend:
    """Kernel backend that runs each session in an isolated Docker container.

    Internet access is blocked via an internal Docker network. The rootfs is
    read-only; the kernel can only write to the bind-mounted workspace and /tmp.

    Configure via environment variables:
        AUTOVIBE_SANDBOX_IMAGE      Docker image (default: autovibe-gym-sandbox:latest)
        AUTOVIBE_KERNEL_MEMORY_MB   Memory cap in MB (default: 2048)
        AUTOVIBE_KERNEL_CPUS        CPU quota string (default: "1")
        AUTOVIBE_KERNEL_PIDS_LIMIT  PID limit (default: 512)
        AUTOVIBE_KERNEL_NETWORK     Internal network name (default: autovibe-kernels)
    """

    def __init__(
        self,
        docker_image: str | None = None,
        memory_mb: int | None = None,
        cpus: str | None = None,
        pids_limit: int | None = None,
        network_name: str | None = None,
    ):
        self.docker_image = docker_image or os.getenv(
            "AUTOVIBE_SANDBOX_IMAGE", "autovibe-gym-sandbox:latest"
        )
        self.memory_mb = _int_env("AUTOVIBE_KERNEL_MEMORY_MB", memory_mb, 2048)
        self.cpus = cpus or os.getenv("AUTOVIBE_KERNEL_CPUS", "1")
        self.pids_limit = _int_env("AUTOVIBE_KERNEL_PIDS_LIMIT", pids_limit, 512)
        self.network_name = network_name or os.getenv(
            "AUTOVIBE_KERNEL_NETWORK", "autovibe-kernels"
        )
        self._ensure_network()

    def create_session(self, workspace_dir: str | Path) -> ContainerJupyterKernelSession:
        return ContainerJupyterKernelSession(
            workspace_dir=workspace_dir,
            docker_image=self.docker_image,
            memory_mb=self.memory_mb,
            cpus=self.cpus,
            pids_limit=self.pids_limit,
            network_name=self.network_name,
        )

    def _ensure_network(self) -> None:
        try:
            subprocess.run(
                [
                    _docker_binary(),
                    "network",
                    "create",
                    "--internal",
                    "--driver",
                    "bridge",
                    self.network_name,
                ],
                capture_output=True,
                timeout=10,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Docker backend is enabled for the kernel but the 'docker' binary "
                "was not found. Install Docker or set DOCKER_BINARY to the correct path."
            ) from exc
        except Exception:
            pass


def _find_free_ports(n: int) -> list[int]:
    """Bind n sockets to ephemeral ports, collect port numbers, then release."""
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            sockets.append(s)
            ports.append(s.getsockname()[1])
    finally:
        for s in sockets:
            s.close()
    return ports


def _docker_binary() -> str:
    return os.getenv("DOCKER_BINARY", "docker")


def _int_env(env_var: str, override: int | None, default: int) -> int:
    if override is not None:
        return override
    raw = os.getenv(env_var)
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


def _output_from_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    try:
        output = output_from_msg(msg)
    except Exception:
        return None
    return output


def _minimal_kernel_env() -> dict[str, str]:
    secret_markers = (
        "KEY",
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "CREDENTIAL",
        "GEMINI",
        "GOOGLE_API",
        "OPENAI",
        "ANTHROPIC",
        "LLM_",
        "MLFLOW_TRACKING",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if value and not any(marker in key.upper() for marker in secret_markers)
    }
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "MPLBACKEND": "Agg",
            "AUTOVIBE_JUPYTER_KERNEL": "1",
        }
    )
    return env

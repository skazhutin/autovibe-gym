from __future__ import annotations

import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from jupyter_client import KernelManager
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


class LocalJupyterKernelBackend:
    def create_session(self, workspace_dir: str | Path) -> "JupyterKernelSession":
        return JupyterKernelSession(workspace_dir=workspace_dir)


class ContainerJupyterKernelBackend:
    """Future backend hook for a container-isolated Jupyter kernel."""

    def create_session(self, workspace_dir: str | Path) -> "JupyterKernelSession":
        raise NotImplementedError(
            "ContainerJupyterKernelBackend is planned but not implemented yet."
        )


class JupyterKernelSession:
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
        if self.km is None:
            return
        try:
            self.km.interrupt_kernel()
        except Exception:
            pass


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

import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


class CodeExecutor:
    """
    Executes LLM-generated Python code with a hard timeout.

    The default backend is Docker. The Docker backend mounts only a temporary
    workspace into the container, passes a tiny allowlisted environment, disables
    networking, and applies container CPU/memory/pid limits. A subprocess backend
    remains available for unit tests and environments where Docker is not present.
    """

    def __init__(
        self,
        timeout: int = 60,
        *,
        backend: str | None = None,
        docker_image: str | None = None,
        memory_limit_mb: int | None = None,
        cpus: str | None = None,
        pids_limit: int | None = None,
        disable_network: bool = True,
        read_only_rootfs: bool = True,
    ):
        self.timeout = timeout
        self.backend = (backend or os.getenv("AUTOVIBE_EXECUTOR_BACKEND", "docker")).lower()
        self.docker_image = docker_image or os.getenv(
            "AUTOVIBE_SANDBOX_IMAGE",
            "autovibe-gym-sandbox:latest",
        )
        self.memory_limit_mb = _int_env("AUTOVIBE_SANDBOX_MEMORY_MB", memory_limit_mb, 2048)
        self.cpus = cpus or os.getenv("AUTOVIBE_SANDBOX_CPUS", "1")
        self.pids_limit = _int_env("AUTOVIBE_SANDBOX_PIDS_LIMIT", pids_limit, 128)
        self.disable_network = disable_network
        self.read_only_rootfs = read_only_rootfs

    def run(self, code: str, namespace: dict | None = None) -> tuple[str, str, dict]:
        ns = dict(namespace) if namespace else {}

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_execution_files(code, ns, tmpdir)
            try:
                if self.backend == "docker":
                    stdout, stderr = self._run_docker(tmpdir)
                elif self.backend == "subprocess":
                    stdout, stderr = self._run_subprocess(tmpdir, paths["runner"])
                else:
                    return "", f"[executor] Unsupported backend: {self.backend}", ns
            except subprocess.TimeoutExpired:
                return "", f"[executor] Step timed out after {self.timeout}s and was killed.", ns
            except FileNotFoundError as exc:
                return "", _docker_missing_message(exc), ns

            output_path = paths["output"]
            if output_path.exists():
                with output_path.open("rb") as f:
                    ns.update(pickle.load(f))
            elif not stderr:
                stderr = "[executor] Process exited without writing output namespace."

        return stdout, stderr, ns

    def _write_execution_files(self, code: str, namespace: dict, tmpdir: str) -> dict[str, Path]:
        root = Path(tmpdir)
        code_path = root / "code.py"
        input_path = root / "ns_in.pkl"
        output_path = root / "ns_out.pkl"
        runner_path = root / "runner.py"

        code_path.write_text(code, encoding="utf-8")
        with input_path.open("wb") as f:
            pickle.dump(self._serialisable(namespace), f)

        if self.backend == "docker":
            policy = self._docker_policy()
            runner_src = self._build_runner(
                "/workspace/code.py",
                "/workspace/ns_in.pkl",
                "/workspace/ns_out.pkl",
                policy,
            )
        else:
            policy = self._subprocess_policy(tmpdir)
            runner_src = self._build_runner(
                str(code_path),
                str(input_path),
                str(output_path),
                policy,
            )
        runner_path.write_text(runner_src, encoding="utf-8")
        return {"code": code_path, "input": input_path, "output": output_path, "runner": runner_path}

    def _run_docker(self, tmpdir: str) -> tuple[str, str]:
        command = self._docker_command(tmpdir)
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout + 5,
        )
        return proc.stdout, proc.stderr

    def _run_subprocess(self, tmpdir: str, runner_path: Path) -> tuple[str, str]:
        proc = subprocess.run(
            [sys.executable, str(runner_path)],
            cwd=tmpdir,
            env=self._sandbox_env(tmpdir),
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return proc.stdout, proc.stderr

    def _docker_command(self, tmpdir: str) -> list[str]:
        command = [
            os.getenv("DOCKER_BINARY", "docker"),
            "run",
            "--rm",
            "--workdir",
            "/workspace",
            "--mount",
            f"type=bind,source={Path(tmpdir).resolve()},target=/workspace",
            "--tmpfs",
            " /tmp:rw,nosuid,nodev,size=256m".strip(),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--cpus",
            str(self.cpus),
            "--memory",
            f"{self.memory_limit_mb}m",
            "--env",
            "PYTHONIOENCODING=utf-8",
            "--env",
            "PYTHONUTF8=1",
            "--env",
            "AUTOVIBE_SANDBOX=1",
            "--env",
            "TMPDIR=/tmp",
            "--env",
            "JOBLIB_MULTIPROCESSING=0",
            "--env",
            "LOKY_MAX_CPU_COUNT=1",
        ]
        if self.disable_network:
            command.extend(["--network", "none"])
        if self.read_only_rootfs:
            command.append("--read-only")
        command.extend(["--entrypoint", "python", self.docker_image, "/workspace/runner.py"])
        return command

    @staticmethod
    def _serialisable(ns: dict) -> dict:
        out = {}
        for key, value in ns.items():
            try:
                pickle.dumps(value)
                out[key] = value
            except Exception:
                pass
        return out

    def _subprocess_policy(self, tmpdir: str) -> dict:
        tmp_path = Path(tmpdir).resolve()
        read_roots = {
            str(tmp_path),
            os.path.abspath(tmpdir),
            str(Path(sys.prefix).resolve()),
            os.path.abspath(sys.prefix),
            str(Path(sys.base_prefix).resolve()),
            os.path.abspath(sys.base_prefix),
        }
        for path in sys.path:
            if not path:
                continue
            try:
                resolved = Path(path).resolve()
            except OSError:
                continue
            if "site-packages" in resolved.parts or "dist-packages" in resolved.parts:
                read_roots.add(str(resolved))
                read_roots.add(os.path.abspath(path))
        return {
            "allowed_read_roots": sorted(read_roots),
            "allowed_write_roots": [str(tmp_path), os.path.abspath(tmpdir)],
            "disable_network": self.disable_network,
            "memory_limit_mb": self.memory_limit_mb,
            "cpu_time_limit_seconds": max(int(self.timeout), 1),
            "python_path_entries": sorted(
                str(Path(path).resolve())
                for path in sys.path
                if path
                and (
                    "site-packages" in Path(path).parts
                    or "dist-packages" in Path(path).parts
                )
            ),
        }

    def _docker_policy(self) -> dict:
        return {
            "allowed_read_roots": [
                "/workspace",
                "/usr/local",
                "/usr/lib",
                "/lib",
                "/opt",
                "/tmp",
            ],
            "allowed_write_roots": ["/workspace", "/tmp"],
            "disable_network": self.disable_network,
            "memory_limit_mb": None,
            "cpu_time_limit_seconds": max(int(self.timeout), 1),
            "python_path_entries": [],
        }

    @staticmethod
    def _sandbox_env(tmpdir: str) -> dict[str, str]:
        keep = {
            "PATH",
            "PATHEXT",
            "SystemRoot",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE",
        }
        env = {key: value for key, value in os.environ.items() if key in keep}
        env.update({
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "TMP": tmpdir,
            "TEMP": tmpdir,
            "TMPDIR": tmpdir,
            "MPLCONFIGDIR": tmpdir,
            "JOBLIB_TEMP_FOLDER": tmpdir,
            "XDG_CACHE_HOME": tmpdir,
            "AUTOVIBE_SANDBOX": "1",
        })
        env.update(thread_limit_env())
        # The sandbox blocks socket and process-spawning syscalls for isolation,
        # so sklearn/joblib n_jobs=-1 (loky/multiprocessing) crashes while
        # starting workers. Force the sequential joblib backend; training then
        # runs in-process and code using n_jobs no longer fails.
        env["JOBLIB_MULTIPROCESSING"] = "0"
        env["LOKY_MAX_CPU_COUNT"] = "1"
        return env

    @staticmethod
    def _build_runner(
        code_path: str,
        input_path: str,
        output_path: str,
        policy: dict,
    ) -> str:
        return textwrap.dedent(f"""
import os
import pickle
import sys

_POLICY = {policy!r}

for _entry in _POLICY.get("python_path_entries", []):
    if _entry and _entry not in sys.path:
        sys.path.insert(0, _entry)


def _path(value):
    if value is None:
        return None
    try:
        return os.path.abspath(os.fspath(value))
    except (TypeError, ValueError, OSError):
        return None


def _under(path, roots):
    path = _path(path)
    if path is None:
        return True
    normalized_path = os.path.normcase(path)
    for root in roots:
        try:
            normalized_root = os.path.normcase(root)
            common = os.path.commonpath([normalized_path, normalized_root])
        except ValueError:
            continue
        if common == normalized_root:
            return True
    return False


def _install_resource_limits():
    if os.name != "posix":
        return
    try:
        import resource
        memory_mb = _POLICY.get("memory_limit_mb")
        if memory_mb:
            limit = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        cpu_seconds = _POLICY.get("cpu_time_limit_seconds")
        if cpu_seconds:
            limit = int(cpu_seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 1))
    except Exception:
        pass


def _install_network_block():
    return


def _install_audit_policy():
    allowed_read = [_path(p) for p in _POLICY["allowed_read_roots"]]
    allowed_write = [_path(p) for p in _POLICY["allowed_write_roots"]]
    allowed_read = [p for p in allowed_read if p]
    allowed_write = [p for p in allowed_write if p]

    def _audit(event, args):
        if event == "open":
            path = args[0] if args else None
            mode = str(args[1] if len(args) > 1 and args[1] is not None else "r")
            writing = any(flag in mode for flag in ("w", "a", "+", "x"))
            if writing and not _under(path, allowed_write):
                raise PermissionError("Sandbox write outside working directory is blocked.")
            if not writing and not _under(path, allowed_read):
                raise PermissionError("Sandbox read outside allowed roots is blocked.")
        elif event.startswith("socket.") and _POLICY.get("disable_network", True):
            raise PermissionError("Network access is disabled in the AutoVibe sandbox.")
        elif event in {{"subprocess.Popen", "os.system", "os.remove", "os.rename", "shutil.rmtree"}}:
            raise PermissionError(f"Sandbox blocked operation: {{event}}")

    sys.addaudithook(_audit)


_install_resource_limits()
_install_network_block()
_install_audit_policy()

with open({input_path!r}, "rb") as f:
    namespace = pickle.load(f)

import numpy as np
import pandas as pd
namespace.setdefault("pd", pd)
namespace.setdefault("np", np)

try:
    exec(compile(open({code_path!r}).read(), "<llm_code>", "exec"), namespace)
except Exception:
    import traceback
    print(traceback.format_exc(), file=sys.stderr)


def _ok(v):
    try:
        pickle.dumps(v)
        return True
    except Exception:
        return False


safe = {{k: v for k, v in namespace.items() if not k.startswith("_") and _ok(v)}}

with open({output_path!r}, "wb") as f:
    pickle.dump(safe, f)
""")


def _int_env(name: str, explicit: int | None, default: int) -> int:
    if explicit is not None:
        return explicit
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


# Native math-library thread-pool caps. On many-core hosts (e.g. the H200
# server) the default per-thread buffer allocation in OpenBLAS/OpenMP can
# exhaust memory and abort sandboxed model training with errors such as
# "OpenBLAS: Memory allocation failed" or xgboost ctypes DataIter crashes.
# Each value is overridable from the parent environment; set
# AUTOVIBE_SANDBOX_THREADS to raise the default for all of them at once.
_THREAD_LIMIT_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def thread_limit_env() -> dict[str, str]:
    """Return thread-cap env vars, respecting any explicit parent override."""
    limit = os.getenv("AUTOVIBE_SANDBOX_THREADS", "1")
    return {var: os.getenv(var, limit) for var in _THREAD_LIMIT_VARS}


def _docker_missing_message(exc: FileNotFoundError) -> str:
    return (
        "[executor] Docker backend is enabled, but the docker CLI was not found. "
        "Install Docker, build Dockerfile.sandbox, or set "
        "AUTOVIBE_EXECUTOR_BACKEND=subprocess for local tests."
    )

"""Run gym episodes on a remote GPU server over SSH while the dashboard stays
local. Primitives: launch (nohup a runner on the server), rsync the episode
workspace + log back, check liveness, kill, and parse the final run summary the
runner prints to stdout.

Auth: SSH key by default (recommended — `ssh-copy-id` once). If a password is
configured it is supplied via an `expect` wrapper (macOS ships `expect`). All
heavy compute (gym + notebook kernel + LLM call) runs on the server; the Mac
only issues ssh/rsync and renders the synced artifacts.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..routers.settings import _load as load_settings

SSH_BASE_OPTS = ["-o", "BatchMode=no", "-o", "ConnectTimeout=12",
                 "-o", "StrictHostKeyChecking=accept-new"]


def config() -> dict[str, Any]:
    s = load_settings()
    return {
        "enabled": bool(s.get("remote_enabled")),
        "ssh": (s.get("remote_ssh") or "").strip(),
        "opts": shlex.split(s.get("remote_ssh_opts") or ""),
        "repo": (s.get("remote_repo") or "").strip(),
        "python": (s.get("remote_python") or "python3").strip(),
        "runs_dir": (s.get("remote_runs_dir") or "~/dash_runs").strip(),
        "password": s.get("remote_password") or "",
    }


def is_enabled() -> bool:
    rc = config()
    return rc["enabled"] and bool(rc["ssh"]) and bool(rc["repo"])


def _expect_wrap(argv: list[str], password: str, timeout: int) -> subprocess.CompletedProcess:
    """Run argv under expect, answering the password prompt."""
    script = (
        f'set timeout {timeout}\n'
        f'spawn {" ".join(shlex.quote(a) for a in argv)}\n'
        'expect {\n'
        '  -re "(?i)password:" { send -- "$env(REMOTE_PW)\\r"; exp_continue }\n'
        '  -re "(?i)passphrase" { send -- "$env(REMOTE_PW)\\r"; exp_continue }\n'
        '  eof\n'
        '}\n'
        'catch wait result\n'
        'exit [lindex $result 3]\n'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".exp", delete=False) as fh:
        fh.write(script)
        exp_path = fh.name
    try:
        return subprocess.run(
            ["expect", "-f", exp_path],
            env={"REMOTE_PW": password, "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
            capture_output=True, text=True, timeout=timeout + 10,
        )
    finally:
        Path(exp_path).unlink(missing_ok=True)


def _run(argv: list[str], rc: dict, timeout: int, stdin: str | None = None) -> subprocess.CompletedProcess:
    if rc["password"]:
        return _expect_wrap(argv, rc["password"], timeout)
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, input=stdin)


def _ssh_argv(rc: dict) -> list[str]:
    return ["ssh", *SSH_BASE_OPTS, *rc["opts"], rc["ssh"]]


def ssh_exec(rc: dict, remote_script: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a bash script on the server (piped via stdin to `bash -s`)."""
    argv = [*_ssh_argv(rc), "bash", "-s"]
    if rc["password"]:
        # expect can't easily also pipe stdin; inline the script via -c instead.
        argv = [*_ssh_argv(rc), "bash", "-lc", remote_script]
        return _run(argv, rc, timeout)
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, input=remote_script)


def check(rc: dict | None = None) -> dict[str, Any]:
    """Connectivity + environment probe for the Settings 'Проверить' button."""
    rc = rc or config()
    if not rc["ssh"]:
        return {"ok": False, "error": "не задан SSH (user@host)"}
    script = (
        f'echo HOST=$(hostname); '
        f'test -d {shlex.quote(rc["repo"])} && echo REPO_OK || echo REPO_MISSING; '
        f'{shlex.quote(rc["python"])} -c "import gym, experiments.run_gym; print(\'GYM_OK\')" 2>&1 | tail -1'
    )
    try:
        res = ssh_exec(rc, script, timeout=25)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"ok": False, "error": str(exc)}
    out = (res.stdout or "") + (res.stderr or "")
    return {
        "ok": res.returncode == 0 and "REPO_OK" in out,
        "returncode": res.returncode,
        "repo": "REPO_OK" in out,
        "gym": "GYM_OK" in out,
        "output": out.strip()[-800:],
    }


def launch(rc: dict, *, run_id: str, runner_args: list[str], env: dict[str, str],
           dataset_rel: str, workspace_remote: str, log_remote: str) -> dict[str, Any]:
    """nohup the runner on the server; return {pid} or {error}."""
    exports = " ".join(f"export {k}={shlex.quote(v)};" for k, v in env.items())
    args = " ".join(shlex.quote(a) for a in runner_args)
    script = (
        f'set -e; mkdir -p {shlex.quote(workspace_remote)}; '
        f'cd {shlex.quote(rc["repo"])}; {exports} '
        f'nohup {shlex.quote(rc["python"])} {args} '
        f'--dataset-dir {shlex.quote(dataset_rel)} '
        f'--workspace-dir {shlex.quote(workspace_remote)} '
        f'> {shlex.quote(log_remote)} 2>&1 & echo PID=$!'
    )
    try:
        res = ssh_exec(rc, script, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"error": str(exc)}
    out = (res.stdout or "") + (res.stderr or "")
    pid = None
    for line in out.splitlines():
        if line.startswith("PID="):
            pid = line.split("=", 1)[1].strip()
    if not pid:
        return {"error": f"launch failed: {out.strip()[-400:]}"}
    return {"pid": pid}


def sync(rc: dict, *, workspace_remote: str, log_remote: str, local_dir: Path) -> bool:
    """rsync the remote workspace + process log into local_dir for parsing."""
    local_dir.mkdir(parents=True, exist_ok=True)
    rsh = "ssh " + " ".join(SSH_BASE_OPTS + rc["opts"])
    specs = [
        (f'{rc["ssh"]}:{workspace_remote}/', str(local_dir / "workspace") + "/"),
        (f'{rc["ssh"]}:{log_remote}', str(local_dir / "process.log")),
    ]
    ok = True
    for remote, local in specs:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        argv = ["rsync", "-az", "--timeout=20", "-e", rsh, remote, local]
        try:
            res = _run(argv, rc, timeout=40)
            ok = ok and res.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            ok = False
    return ok


def alive(rc: dict, pid: str) -> bool:
    if not pid:
        return False
    try:
        res = ssh_exec(rc, f'kill -0 {int(pid)} 2>/dev/null && echo ALIVE || echo DEAD', timeout=20)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return False
    return "ALIVE" in (res.stdout or "")


def kill(rc: dict, pid: str) -> None:
    if not pid:
        return
    try:
        ssh_exec(rc, f'kill -TERM -{int(pid)} 2>/dev/null; kill -TERM {int(pid)} 2>/dev/null; true', timeout=20)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass


def parse_summary(log_text: str) -> dict[str, Any] | None:
    """Extract the JSON run summary run_gym prints after '=== Run Summary ==='."""
    marker = "=== Run Summary ==="
    idx = log_text.rfind(marker)
    if idx == -1:
        return None
    tail = log_text[idx + len(marker):]
    start = tail.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(tail[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(tail[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None

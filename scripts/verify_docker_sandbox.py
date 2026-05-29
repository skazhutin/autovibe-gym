"""
Smoke test for the Docker kernel sandbox (ContainerJupyterKernelBackend).

Verifies three properties:
  1. Kernel starts and responds to code execution
  2. Bootstrap context (train_df / val_df) loads correctly
  3. Outbound network access is blocked inside the container

Run after building the sandbox image:
    docker build -f Dockerfile.sandbox -t autovibe-gym-sandbox:latest .
    python scripts/verify_docker_sandbox.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def _check_docker() -> None:
    import subprocess
    result = subprocess.run(
        ["docker", "info"], capture_output=True, timeout=10
    )
    if result.returncode != 0:
        print("ERROR: Docker daemon is not running or not accessible.", file=sys.stderr)
        sys.exit(1)


def _write_data(workspace: Path) -> tuple[Path, Path]:
    import pandas as pd

    data_dir = workspace / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0],
        "x2": [0.1, 0.2, 0.3, 0.4],
        "target": [0, 1, 0, 1],
    })
    val = pd.DataFrame({
        "x1": [5.0, 6.0],
        "x2": [0.5, 0.6],
        "target": [1, 0],
    })
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    return train_path, val_path


def main() -> None:
    print("AutoVibe Gym — Docker sandbox verification")
    print("=" * 50)

    _check_docker()
    print("[1/4] Docker daemon: OK")

    from gym.jupyter_kernel import ContainerJupyterKernelBackend, ContainerJupyterKernelSession

    with tempfile.TemporaryDirectory(prefix="autovibe_verify_") as tmpdir:
        workspace = Path(tmpdir)
        train_path, val_path = _write_data(workspace)

        backend = ContainerJupyterKernelBackend(
            docker_image=os.getenv("AUTOVIBE_SANDBOX_IMAGE", "autovibe-gym-sandbox:latest"),
        )
        session: ContainerJupyterKernelSession = backend.create_session(workspace)

        print("[2/4] Starting kernel container…")
        session.start()
        print("      Container ID:", session._container_id)

        try:
            # Test 1: bootstrap context
            session.inject_bootstrap_context(
                train_csv=train_path,
                val_csv=val_path,
                target_col="target",
            )
            result = session.execute_cell("print(train_df.shape, val_df.shape, target_col)")
            assert result.success, f"Bootstrap failed: {result.error_name}: {result.error_value}"
            assert "(4, 3)" in result.stdout, f"Unexpected output: {result.stdout!r}"
            print("[3/4] Bootstrap + code execution: OK  →", result.stdout.strip())

            # Test 2: network blocked
            net_result = session.execute_cell(
                "import socket\n"
                "try:\n"
                "    socket.create_connection(('8.8.8.8', 53), timeout=2)\n"
                "    print('CONNECTED')\n"
                "except OSError:\n"
                "    print('BLOCKED')\n"
            )
            assert "BLOCKED" in net_result.stdout, (
                "Network should be blocked inside container! "
                f"Got stdout={net_result.stdout!r}"
            )
            print("[4/4] Outbound network: BLOCKED (correct)  →", net_result.stdout.strip())

        finally:
            session.shutdown()
            print("      Container stopped.")

    print()
    print("All checks passed. Docker sandbox is working correctly.")


if __name__ == "__main__":
    main()

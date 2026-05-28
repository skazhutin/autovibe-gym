import os
import pickle
import subprocess
import sys
import tempfile
import textwrap


class CodeExecutor:
    """
    Executes LLM-generated Python code in a subprocess with a hard timeout.
    Namespace is passed in/out via pickle tempfiles.
    Test data is never injected — only train_df and val_df are available.
    """

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def run(self, code: str, namespace: dict | None = None) -> tuple[str, str, dict]:
        ns = dict(namespace) if namespace else {}

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "code.py")
            input_path = os.path.join(tmpdir, "ns_in.pkl")
            output_path = os.path.join(tmpdir, "ns_out.pkl")
            runner_path = os.path.join(tmpdir, "runner.py")

            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)

            with open(input_path, "wb") as f:
                pickle.dump(self._serialisable(ns), f)

            runner_src = self._build_runner(code_path, input_path, output_path)
            with open(runner_path, "w", encoding="utf-8") as f:
                f.write(runner_src)

            try:
                proc = subprocess.run(
                    [sys.executable, runner_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                )
                stdout = proc.stdout
                stderr = proc.stderr

                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        updated_ns = pickle.load(f)
                    # Merge: keep original values that survived, plus new ones
                    ns.update(updated_ns)
                else:
                    # Subprocess crashed before writing output
                    if not stderr:
                        stderr = "[executor] Process exited without writing output namespace."

            except subprocess.TimeoutExpired:
                stdout = ""
                stderr = f"[executor] Step timed out after {self.timeout}s and was killed."

        return stdout, stderr, ns

    @staticmethod
    def _serialisable(ns: dict) -> dict:
        out = {}
        for k, v in ns.items():
            try:
                pickle.dumps(v)
                out[k] = v
            except Exception:
                pass
        return out

    @staticmethod
    def _build_runner(code_path: str, input_path: str, output_path: str) -> str:
        return textwrap.dedent(f"""
import pickle, sys

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

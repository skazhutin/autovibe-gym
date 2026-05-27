import io
import sys
import traceback


class CodeExecutor:
    """
    Executes LLM-generated Python code in an isolated namespace.
    Returns (stdout, stderr, updated_namespace).
    Test data is never injected — only train_df and val_df are available.
    """

    def run(
        self, code: str, namespace: dict | None = None
    ) -> tuple[str, str, dict]:
        ns = dict(namespace) if namespace else {}

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_buf, stderr_buf

        try:
            exec(compile(code, "<llm_code>", "exec"), ns)
            stderr_out = ""
        except Exception:
            stderr_out = traceback.format_exc()
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        stdout_out = stdout_buf.getvalue()
        if not stderr_out:
            stderr_out = stderr_buf.getvalue()

        # Strip private keys so the LLM can't inspect env internals
        safe_ns = {k: v for k, v in ns.items() if not k.startswith("_")}
        return stdout_out, stderr_out, safe_ns

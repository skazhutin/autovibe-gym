"""Post-run solution summary.

After an episode finishes, we make ONE extra LLM call that asks the model to
summarize the solution it just produced. The result is saved as
``run_summary.json`` in the episode workspace and rendered at the top of the
dashboard «Мысли» tab — for every run, even those without thoughts mode.

Privacy: the summary is built only from the conversation/context the model
already saw during the run. The hidden test score is never part of that
context, so asking the model to summarize cannot leak it. We additionally never
feed the score into the summary request here.

The call is best-effort: any client/provider error (rate limit, network, etc.)
is swallowed and the run proceeds without a summary rather than failing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

SUMMARY_FILENAME = "run_summary.json"

# A weak model follows English instructions more reliably; we still ask for a
# Russian summary because the whole dashboard audience is Russian-speaking.
SUMMARY_SYSTEM_PROMPT = """The machine-learning episode is over. Write a short,
well-structured summary of YOUR OWN solution, in Russian, for your teammates.

Use compact markdown with exactly these sections:
- **Подход** — which model / pipeline you chose and why.
- **Ключевые шаги** — preprocessing, features and validation you actually did.
- **Результат** — what you achieved on validation and whether the submission went through.
- **Что улучшить** — 1-3 concrete next ideas.

Rules:
- Base everything ONLY on what actually happened in this episode. Do not invent
  numbers or steps you never took.
- Maximum ~180 words. No code blocks — just the gist.
- Output only the summary text as markdown. No JSON, no preamble, no headings
  other than the bold section labels above."""

SUMMARY_REQUEST = (
    "Эпизод завершён. Напиши финальное саммари своего решения строго по "
    "структуре из системного промпта (Подход / Ключевые шаги / Результат / "
    "Что улучшить). Верни только текст саммари в markdown."
)

# Keep the post-run call cheap and within tight per-minute token budgets.
DEFAULT_MAX_TOKENS = 700
# How many trailing conversation messages to keep (plus the first task message)
# so the summary stays grounded in the final solution without blowing context.
DEFAULT_MAX_MESSAGES = 16


def _trim_conversation(
    conversation: list[dict[str, Any]], max_messages: int
) -> list[dict[str, Any]]:
    """Keep the opening task message plus the most recent exchanges."""
    if max_messages <= 0 or len(conversation) <= max_messages:
        return list(conversation)
    head = conversation[:1]
    tail = conversation[-(max_messages - len(head)):]
    return head + tail


def generate_summary(
    client: Any,
    model: str,
    *,
    conversation: list[dict[str, Any]] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_messages: int = DEFAULT_MAX_MESSAGES,
) -> dict[str, Any] | None:
    """Make one best-effort LLM call summarizing the finished solution.

    ``conversation`` is the message list the model worked with (task prompt,
    its own actions, environment feedback). We append the summary request and
    ask the model to reflect. Returns a JSON-serializable payload, or ``None``
    if the call failed or produced nothing.
    """
    if client is None or not model:
        return None
    messages = _trim_conversation(list(conversation or []), max_messages)
    messages.append({"role": "user", "content": SUMMARY_REQUEST})
    try:
        response = client.complete(
            model=model,
            max_tokens=max(64, int(max_tokens)),
            system=SUMMARY_SYSTEM_PROMPT,
            messages=messages,
        )
    except Exception:
        # Best-effort: a summary must never break or fail a finished run.
        return None
    text = (getattr(response, "text", "") or "").strip()
    if not text:
        return None
    return {
        "summary": text,
        "model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_tokens": int(getattr(response, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(response, "output_tokens", 0) or 0),
    }


def write_summary(workspace_dir: str | Path | None, payload: dict[str, Any] | None) -> None:
    """Persist the summary payload as run_summary.json in the episode workspace."""
    if not payload or not workspace_dir:
        return
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / SUMMARY_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), "utf-8"
    )


def generate_and_write(
    client: Any,
    model: str,
    workspace_dir: str | Path | None,
    *,
    conversation: list[dict[str, Any]] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_messages: int = DEFAULT_MAX_MESSAGES,
) -> dict[str, Any] | None:
    """Convenience: generate the summary and write it to the workspace."""
    payload = generate_summary(
        client,
        model,
        conversation=conversation,
        max_tokens=max_tokens,
        max_messages=max_messages,
    )
    write_summary(workspace_dir, payload)
    return payload

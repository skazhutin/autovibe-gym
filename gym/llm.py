import os
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


def _message_text(message) -> str:
    """Extract the assistant text. Some models (gpt-oss/harmony, o1-style) leave
    `content` empty and put the answer in a `reasoning` field — fall back to it."""
    content = getattr(message, "content", None)
    if content:
        return content
    reasoning = getattr(message, "reasoning", None)
    if not reasoning:
        extra = getattr(message, "model_extra", None) or {}
        reasoning = extra.get("reasoning") or extra.get("reasoning_content")
    return reasoning or ""


class LLMClient(Protocol):
    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        ...


_TRANSIENT_NAMES = {
    "RateLimitError", "APITimeoutError", "APIConnectionError",
    "InternalServerError", "APIError",
}
_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def _is_transient(exc: Exception) -> bool:
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return status in _TRANSIENT_STATUS


def _create_with_retries(client, **kwargs):
    """Call chat.completions.create with exponential backoff on transient
    errors (rate limits / 5xx / timeouts). Tunable via env:
    AUTOVIBE_LLM_MAX_RETRIES (default 5), AUTOVIBE_LLM_RETRY_BASE (default 2s)."""
    try:
        max_retries = int(os.getenv("AUTOVIBE_LLM_MAX_RETRIES", "5"))
    except ValueError:
        max_retries = 5
    try:
        base = float(os.getenv("AUTOVIBE_LLM_RETRY_BASE", "2"))
    except ValueError:
        base = 2.0
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - retry only on transient errors
            if attempt >= max_retries or not _is_transient(exc):
                raise
            delay = min(base * (2 ** attempt), 30.0)
            print(f"[llm] transient error ({type(exc).__name__}); retry "
                  f"{attempt + 1}/{max_retries} in {delay:.0f}s", flush=True)
            time.sleep(delay)


class LiteLLMClient:
    """
    Adapter that uses the LiteLLM Python SDK directly — no proxy server needed.

    Supports any provider litellm knows about. Configure via env vars:
      LLM_MODEL   — e.g. "groq/llama-3.3-70b-versatile"
      GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.

    See: https://docs.litellm.ai/docs/providers
    """

    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        import litellm

        response = litellm.completion(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
        )
        usage = response.usage
        text = _message_text(response.choices[0].message)
        return LLMResponse(
            text=text.strip(),
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )


class OpenAICompatibleLLMClient:
    """
    Adapter for OpenAI-compatible chat APIs.

    Works with local vLLM, OpenAI, and proxy servers configured through:
    LLM_BASE_URL, LLM_API_KEY.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        # Explicit per-request timeout so a hanging/queued request (e.g. several
        # concurrent runs against one rate-limited endpoint) fails fast as an
        # APITimeoutError and is retried with backoff, instead of silently
        # blocking for the SDK default (~10 min). max_retries=0 so OUR
        # _create_with_retries owns retrying. Tunable via AUTOVIBE_LLM_TIMEOUT.
        try:
            timeout = float(os.getenv("AUTOVIBE_LLM_TIMEOUT", "120"))
        except ValueError:
            timeout = 120.0
        self._client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.getenv("LLM_API_KEY", "local"),
            timeout=timeout,
            max_retries=0,
        )

    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        response = _create_with_retries(
            self._client,
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
        )
        usage = response.usage
        text = _message_text(response.choices[0].message)
        return LLMResponse(
            text=text.strip(),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


class GoogleAIStudioLLMClient:
    """
    Adapter for Google AI Studio / Gemini API through the google-genai SDK.

    Configure with GEMINI_API_KEY or GOOGLE_API_KEY. This client is only
    imported when LLM_PROVIDER=google or LLM_PROVIDER=gemini is selected.
    """

    def __init__(self, api_key: str | None = None):
        from google import genai

        resolved_api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError(
                "Google AI provider requires GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        self._client = genai.Client(api_key=resolved_api_key)
        self._last_request_at = 0.0

    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        from google.genai import types

        _wait_for_min_request_interval(self)
        response = _call_with_retries(
            lambda: self._client.models.generate_content(
                model=model,
                contents=_messages_to_google_contents(messages),
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                ),
            )
        )
        self._last_request_at = time.monotonic()
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=_google_response_text(response),
            input_tokens=_usage_count(usage, "prompt_token_count"),
            output_tokens=_usage_count(usage, "candidates_token_count"),
        )


def make_llm_client(provider: str | None = None) -> LLMClient:
    normalized = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    if normalized in {"openai", "openai-compatible", "openai_compatible", "vllm"}:
        return OpenAICompatibleLLMClient()
    if normalized in {"google", "google-ai", "google_ai", "gemini"}:
        return GoogleAIStudioLLMClient()
    if normalized in {"litellm", "lite-llm", "groq", "anthropic"}:
        return LiteLLMClient()
    raise ValueError(
        f"Unsupported LLM_PROVIDER={normalized!r}. Use 'openai', 'google', or 'litellm'."
    )


def default_model_name(provider: str | None = None) -> str:
    if os.getenv("LLM_MODEL"):
        return os.getenv("LLM_MODEL", "")
    normalized = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    if normalized in {"google", "google-ai", "google_ai", "gemini"}:
        return "gemini-2.5-flash"
    if normalized in {"litellm", "lite-llm", "groq"}:
        return "groq/llama-3.3-70b-versatile"
    return "Qwen/Qwen2.5-Coder-7B-Instruct"


def _messages_to_google_contents(messages: list[dict]) -> list[dict]:
    contents = []
    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = str(message.get("content", ""))
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": content}],
            }
        )
    return contents


def _usage_count(usage: object | None, attr: str) -> int:
    value = getattr(usage, attr, 0) if usage else 0
    return int(value or 0)


def _google_response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    parts_text: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts_text.append(str(part_text))
    return "".join(parts_text).strip()


def _call_with_retries(operation):
    attempts = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
    delay = float(os.getenv("LLM_RETRY_INITIAL_DELAY", "2"))
    last_error = None

    for attempt in range(max(attempts, 1)):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt == attempts - 1 or not _is_transient_error(exc):
                raise
            time.sleep(delay)
            delay *= 2

    raise last_error


def _is_transient_error(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        name = current.__class__.__name__.lower()
        if any(token in name for token in ("timeout", "connecterror", "networkerror")):
            return True
        status_code = getattr(current, "status_code", None) or getattr(
            current, "code", None
        )
        if status_code in {
            HTTPStatus.TOO_MANY_REQUESTS,
            HTTPStatus.INTERNAL_SERVER_ERROR,
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.SERVICE_UNAVAILABLE,
            HTTPStatus.GATEWAY_TIMEOUT,
        }:
            return True
        current = current.__cause__ or current.__context__
    return False


def _wait_for_min_request_interval(client: object) -> None:
    interval = float(os.getenv("LLM_MIN_REQUEST_INTERVAL_SECONDS", "0") or "0")
    if interval <= 0:
        return

    last_request_at = getattr(client, "_last_request_at", 0.0)
    elapsed = time.monotonic() - last_request_at
    remaining = interval - elapsed
    if last_request_at and remaining > 0:
        time.sleep(remaining)

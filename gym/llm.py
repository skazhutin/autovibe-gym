import os
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Callable, Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0


def _openai_usage_counts(usage: object | None) -> tuple[int, int, int, int]:
    if usage is None:
        return 0, 0, 0, 0
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        max(completion_tokens - reasoning_tokens, 0),
        reasoning_tokens,
        int(getattr(prompt_details, "cached_tokens", 0) or 0),
    )


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


def configured_temperature() -> float:
    raw = os.getenv("AUTOVIBE_LLM_TEMPERATURE", "1.0")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("AUTOVIBE_LLM_TEMPERATURE must be numeric") from exc
    if not 0.0 <= value <= 2.0:
        raise ValueError("AUTOVIBE_LLM_TEMPERATURE must be between 0 and 2")
    return value


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


def _emit_attempt(hook: Callable | None, **event) -> None:
    if hook is None:
        return
    try:
        hook(event)
    except Exception:
        # Research observability must never alter provider behavior.
        return


def _create_with_retries(client, *, request_attempt_hook=None, **kwargs):
    """Call chat.completions.create with exponential backoff on transient
    errors (rate limits / 5xx / timeouts). Tunable via env:
    AUTOVIBE_LLM_MAX_RETRIES (default 3), AUTOVIBE_LLM_RETRY_BASE (default 2s)."""
    try:
        max_retries = int(os.getenv("AUTOVIBE_LLM_MAX_RETRIES", "3"))
    except ValueError:
        max_retries = 3
    try:
        base = float(os.getenv("AUTOVIBE_LLM_RETRY_BASE", "2"))
    except ValueError:
        base = 2.0
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - retry only on transient errors
            _emit_attempt(
                request_attempt_hook,
                retry_index=attempt,
                success=False,
                duration_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
                http_status=getattr(exc, "status_code", None) or getattr(exc, "code", None),
                request_id=getattr(exc, "request_id", None),
            )
            if attempt >= max_retries or not _is_transient(exc):
                raise
            delay = min(base * (2 ** attempt), 30.0)
            print(f"[llm] transient error ({type(exc).__name__}); retry "
                  f"{attempt + 1}/{max_retries} in {delay:.0f}s", flush=True)
            time.sleep(delay)
        else:
            usage = getattr(response, "usage", None)
            input_tokens, output_tokens, reasoning_tokens, cached_input_tokens = (
                _openai_usage_counts(usage)
            )
            _emit_attempt(
                request_attempt_hook,
                retry_index=attempt,
                success=True,
                duration_seconds=time.perf_counter() - started,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_input_tokens=cached_input_tokens,
                request_id=getattr(response, "id", None),
                finish_reason=getattr(response.choices[0], "finish_reason", None),
            )
            return response


class LiteLLMClient:
    """
    Adapter that uses the LiteLLM Python SDK directly — no proxy server needed.

    Supports any provider litellm knows about. The shared model registry passes
    the selected model name (e.g. "groq/llama-3.3-70b-versatile") and optional
    per-model API key through AUTOVIBE_LITELLM_API_KEY.

    See: https://docs.litellm.ai/docs/providers
    """

    def __init__(self):
        self._research_attempt_hook = None

    def set_request_attempt_hook(self, hook: Callable | None) -> None:
        self._research_attempt_hook = hook

    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        import litellm

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": configured_temperature(),
            "messages": [{"role": "system", "content": system}] + messages,
        }
        api_key = os.getenv("AUTOVIBE_LITELLM_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        started = time.perf_counter()
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            _emit_attempt(
                self._research_attempt_hook,
                retry_index=0,
                success=False,
                duration_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
                http_status=getattr(exc, "status_code", None) or getattr(exc, "code", None),
                request_id=getattr(exc, "request_id", None),
            )
            raise
        usage = response.usage
        input_tokens, output_tokens, reasoning_tokens, cached_input_tokens = (
            _openai_usage_counts(usage)
        )
        _emit_attempt(
            self._research_attempt_hook,
            retry_index=0,
            success=True,
            duration_seconds=time.perf_counter() - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_input_tokens=cached_input_tokens,
            request_id=getattr(response, "id", None),
            finish_reason=getattr(response.choices[0], "finish_reason", None),
        )
        text = _message_text(response.choices[0].message)
        return LLMResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_input_tokens=cached_input_tokens,
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
        self._research_attempt_hook = None

    def set_request_attempt_hook(self, hook: Callable | None) -> None:
        self._research_attempt_hook = hook

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
            request_attempt_hook=self._research_attempt_hook,
            model=model,
            max_tokens=max_tokens,
            temperature=configured_temperature(),
            messages=[{"role": "system", "content": system}] + messages,
        )
        usage = response.usage
        input_tokens, output_tokens, reasoning_tokens, cached_input_tokens = (
            _openai_usage_counts(usage)
        )
        text = _message_text(response.choices[0].message)
        return LLMResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_input_tokens=cached_input_tokens,
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
        self._research_attempt_hook = None

    def set_request_attempt_hook(self, hook: Callable | None) -> None:
        self._research_attempt_hook = hook

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
                    temperature=configured_temperature(),
                ),
            ),
            request_attempt_hook=self._research_attempt_hook,
        )
        self._last_request_at = time.monotonic()
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=_google_response_text(response),
            input_tokens=_usage_count(usage, "prompt_token_count"),
            output_tokens=_usage_count(usage, "candidates_token_count"),
            reasoning_tokens=_usage_count(usage, "thoughts_token_count"),
            cached_input_tokens=_usage_count(usage, "cached_content_token_count"),
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


def _call_with_retries(operation, *, request_attempt_hook=None):
    attempts = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
    delay = float(os.getenv("LLM_RETRY_INITIAL_DELAY", "2"))
    last_error = None

    for attempt in range(max(attempts, 1)):
        started = time.perf_counter()
        try:
            response = operation()
        except Exception as exc:
            last_error = exc
            _emit_attempt(
                request_attempt_hook,
                retry_index=attempt,
                success=False,
                duration_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
                http_status=getattr(exc, "status_code", None) or getattr(exc, "code", None),
                request_id=getattr(exc, "request_id", None),
            )
            if attempt == attempts - 1 or not _is_transient_error(exc):
                raise
            time.sleep(delay)
            delay *= 2
        else:
            usage = getattr(response, "usage_metadata", None)
            _emit_attempt(
                request_attempt_hook,
                retry_index=attempt,
                success=True,
                duration_seconds=time.perf_counter() - started,
                input_tokens=_usage_count(usage, "prompt_token_count"),
                output_tokens=_usage_count(usage, "candidates_token_count"),
                reasoning_tokens=_usage_count(usage, "thoughts_token_count"),
                cached_input_tokens=_usage_count(usage, "cached_content_token_count"),
                request_id=getattr(response, "response_id", None),
                finish_reason=None,
            )
            return response

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

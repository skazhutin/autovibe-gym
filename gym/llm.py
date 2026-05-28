import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


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
        text = response.choices[0].message.content or ""
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

        self._client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.getenv("LLM_API_KEY", "local"),
        )

    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
        )
        usage = response.usage
        text = response.choices[0].message.content or ""
        return LLMResponse(
            text=text.strip(),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

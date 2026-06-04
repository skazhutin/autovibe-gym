import sys
import types

import pytest

from gym.llm import (
    GoogleAIStudioLLMClient,
    LLMResponse,
    OpenAICompatibleLLMClient,
    _call_with_retries,
    _google_response_text,
    _is_transient_error,
    _messages_to_google_contents,
    _usage_count,
    _wait_for_min_request_interval,
    default_model_name,
    make_llm_client,
)


class FakeCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        message = types.SimpleNamespace(content='{"type": "submit"}')
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(usage=usage, choices=[choice])


class FakeOpenAI:
    last_instance = None

    def __init__(self, *, base_url, api_key, timeout=None, max_retries=None, **_kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.chat = types.SimpleNamespace(completions=FakeCompletions())
        FakeOpenAI.last_instance = self


def test_openai_compatible_client_uses_env_and_prepends_system_message(monkeypatch):
    fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("LLM_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    client = OpenAICompatibleLLMClient()

    response = client.complete(
        system="system prompt",
        messages=[{"role": "user", "content": "hello"}],
        model="model-a",
        max_tokens=99,
    )

    assert FakeOpenAI.last_instance.base_url == "http://example.test/v1"
    assert FakeOpenAI.last_instance.api_key == "secret"
    request = FakeOpenAI.last_instance.chat.completions.requests[0]
    assert request["model"] == "model-a"
    assert request["max_tokens"] == 99
    assert request["messages"][0] == {"role": "system", "content": "system prompt"}
    assert response == LLMResponse(
        text='{"type": "submit"}',
        input_tokens=11,
        output_tokens=7,
    )


def test_llm_response_defaults_to_zero_token_counts():
    assert LLMResponse(text="ok").input_tokens == 0
    assert LLMResponse(text="ok").output_tokens == 0


def test_make_llm_client_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr("gym.llm.OpenAICompatibleLLMClient", lambda: "openai-client")

    assert make_llm_client() == "openai-client"


def test_make_llm_client_can_select_google(monkeypatch):
    monkeypatch.setattr("gym.llm.GoogleAIStudioLLMClient", lambda: "google-client")

    assert make_llm_client("google") == "google-client"
    assert make_llm_client("gemini") == "google-client"


def test_make_llm_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        make_llm_client("unknown")


def test_default_model_name_keeps_existing_default_for_openai(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    assert default_model_name() == "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_default_model_name_uses_gemini_default_for_google(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert default_model_name("google") == "gemini-2.5-flash"


def test_default_model_name_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "custom-model")

    assert default_model_name("google") == "custom-model"


def test_google_message_conversion_maps_assistant_to_model_role():
    contents = _messages_to_google_contents(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    )

    assert contents == [
        {"role": "user", "parts": [{"text": "question"}]},
        {"role": "model", "parts": [{"text": "answer"}]},
    ]


def test_google_response_text_falls_back_to_candidate_parts():
    part = types.SimpleNamespace(text="candidate text")
    content = types.SimpleNamespace(parts=[part])
    candidate = types.SimpleNamespace(content=content)
    response = types.SimpleNamespace(text="", candidates=[candidate])

    assert _google_response_text(response) == "candidate text"


def test_usage_count_treats_missing_or_none_as_zero():
    usage = types.SimpleNamespace(prompt_token_count=None)

    assert _usage_count(usage, "prompt_token_count") == 0
    assert _usage_count(None, "prompt_token_count") == 0


def test_retry_helper_retries_transient_errors(monkeypatch):
    calls = {"count": 0}

    class ConnectError(Exception):
        pass

    def operation():
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConnectError("temporary")
        return "ok"

    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "2")
    monkeypatch.setattr("gym.llm.time.sleep", lambda delay: None)

    assert _call_with_retries(operation) == "ok"
    assert calls["count"] == 2


def test_transient_error_detection_checks_status_code():
    error = RuntimeError("service unavailable")
    error.status_code = 503

    assert _is_transient_error(error)


def test_min_request_interval_sleeps_between_calls(monkeypatch):
    sleeps = []
    client = types.SimpleNamespace(_last_request_at=100.0)
    monkeypatch.setenv("LLM_MIN_REQUEST_INTERVAL_SECONDS", "10")
    monkeypatch.setattr("gym.llm.time.monotonic", lambda: 103.0)
    monkeypatch.setattr("gym.llm.time.sleep", lambda delay: sleeps.append(delay))

    _wait_for_min_request_interval(client)

    assert sleeps == [7.0]


def test_google_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "google",
        types.SimpleNamespace(genai=types.SimpleNamespace(Client=lambda **kwargs: None)),
    )

    with pytest.raises(ValueError, match="requires GEMINI_API_KEY"):
        GoogleAIStudioLLMClient()


def test_google_client_generates_content_with_system_instruction(monkeypatch):
    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModels:
        def __init__(self):
            self.requests = []

        def generate_content(self, **kwargs):
            self.requests.append(kwargs)
            usage = types.SimpleNamespace(prompt_token_count=13, candidates_token_count=8)
            return types.SimpleNamespace(text=" response ", usage_metadata=usage)

    class FakeGenAIClient:
        last_instance = None

        def __init__(self, *, api_key):
            self.api_key = api_key
            self.models = FakeModels()
            FakeGenAIClient.last_instance = self

    fake_genai_module = types.ModuleType("google.genai")
    fake_genai_module.Client = FakeGenAIClient
    fake_genai_module.types = types.SimpleNamespace(
        GenerateContentConfig=FakeGenerateContentConfig
    )
    fake_google_module = types.ModuleType("google")
    fake_google_module.genai = fake_genai_module
    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)
    monkeypatch.setenv("GEMINI_API_KEY", "token")

    client = GoogleAIStudioLLMClient()
    response = client.complete(
        system="system",
        messages=[{"role": "user", "content": "hello"}],
        model="gemini-test",
        max_tokens=77,
    )

    assert FakeGenAIClient.last_instance.api_key == "token"
    request = FakeGenAIClient.last_instance.models.requests[0]
    assert request["model"] == "gemini-test"
    assert request["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert request["config"].kwargs == {
        "system_instruction": "system",
        "max_output_tokens": 77,
    }
    assert response == LLMResponse(text="response", input_tokens=13, output_tokens=8)

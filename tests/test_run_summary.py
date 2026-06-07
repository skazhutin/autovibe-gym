import json

import pytest

from gym import run_summary
from gym.llm import LLMResponse


class FakeClient:
    def __init__(self, text="**Подход** — лес.\n- шаг 1\n- шаг 2", *, raises=False):
        self.text = text
        self.raises = raises
        self.calls = []

    def complete(self, *, system, messages, model, max_tokens):
        self.calls.append({"system": system, "messages": list(messages),
                           "model": model, "max_tokens": max_tokens})
        if self.raises:
            raise RuntimeError("provider 429")
        return LLMResponse(text=self.text, input_tokens=11, output_tokens=7)


def test_generate_summary_appends_request_and_keeps_system_prompt():
    client = FakeClient()
    convo = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "my solution code"},
    ]
    out = run_summary.generate_summary(client, "m", conversation=convo, max_tokens=300)

    assert out["summary"].startswith("**Подход**")
    assert out["model"] == "m"
    assert out["input_tokens"] == 11 and out["output_tokens"] == 7
    assert out["generated_at"]
    # Exactly the conversation we passed + the summary request — nothing else is
    # injected, so a hidden score that was never in the conversation cannot leak.
    sent = client.calls[0]["messages"]
    assert sent[:2] == convo
    assert sent[-1]["content"] == run_summary.SUMMARY_REQUEST
    assert client.calls[0]["system"] == run_summary.SUMMARY_SYSTEM_PROMPT
    assert client.calls[0]["max_tokens"] == 300


def test_generate_summary_does_not_carry_hidden_score():
    """The summarizer only ever sees the conversation it is handed; it never
    fabricates or forwards a hidden test metric on its own."""
    client = FakeClient()
    convo = [{"role": "assistant", "content": "trained a pipeline on raw rows"}]
    run_summary.generate_summary(client, "m", conversation=convo)
    blob = json.dumps(client.calls[0], ensure_ascii=False)
    assert "test_metric" not in blob
    assert "final_test_metric" not in blob


def test_generate_summary_is_best_effort_on_client_error():
    client = FakeClient(raises=True)
    assert run_summary.generate_summary(client, "m", conversation=[]) is None


def test_generate_summary_returns_none_on_empty_text():
    assert run_summary.generate_summary(FakeClient(text="   "), "m", conversation=[]) is None
    assert run_summary.generate_summary(None, "m", conversation=[]) is None
    assert run_summary.generate_summary(FakeClient(), "", conversation=[]) is None


def test_trim_conversation_keeps_first_and_recent():
    convo = [{"role": "user", "content": f"m{i}"} for i in range(40)]
    trimmed = run_summary._trim_conversation(convo, 5)
    assert len(trimmed) == 5
    assert trimmed[0] == convo[0]            # opening task message kept
    assert trimmed[-1] == convo[-1]          # most recent kept
    # Short conversations pass through unchanged.
    assert run_summary._trim_conversation(convo[:3], 5) == convo[:3]


def test_write_summary_and_generate_and_write(tmp_path):
    run_summary.write_summary(tmp_path, None)  # no payload → no file
    assert not (tmp_path / run_summary.SUMMARY_FILENAME).exists()

    payload = run_summary.generate_and_write(
        FakeClient(), "m", tmp_path, conversation=[{"role": "user", "content": "t"}]
    )
    p = tmp_path / run_summary.SUMMARY_FILENAME
    assert p.exists()
    saved = json.loads(p.read_text("utf-8"))
    assert saved["summary"] == payload["summary"]
    assert saved["model"] == "m"


def test_generate_and_write_failed_call_writes_nothing(tmp_path):
    out = run_summary.generate_and_write(FakeClient(raises=True), "m", tmp_path, conversation=[])
    assert out is None
    assert not (tmp_path / run_summary.SUMMARY_FILENAME).exists()

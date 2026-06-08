from __future__ import annotations

import json
import os

from .env import GymEnv
from .llm import LLMClient, default_model_name, make_llm_client
from .notebook_env import NotebookGymEnv
from .prompts import (
    DEFAULT_THOUGHTS_OFF,
    DEFAULT_THOUGHTS_ON,
    assemble_body,
    build_system_prompt,
)

from .protocol import ACTION_JSON_SCHEMA, Action, ActionParseError, Observation


def _default_client() -> LLMClient:
    """Backward-compatible helper used by older experiment scripts."""
    return make_llm_client()


# Canonical source of truth is gym.prompts. The names below are kept as
# re-exports because experiments/run_fixed.py and the existing test suite
# import them directly. With no overrides, byte-identical to the historical
# constants — guarded by tests/test_prompts.py.
SYSTEM_PROMPT: str = assemble_body()
THOUGHTS_ENABLED_PROMPT: str = DEFAULT_THOUGHTS_ON
THOUGHTS_DISABLED_PROMPT: str = DEFAULT_THOUGHTS_OFF


class GymAgent:
    """
    LLM agent that interacts with Gym environments through explicit JSON actions.

    The default client is selected by `make_llm_client()` and can target
    OpenAI-compatible APIs, Google AI Studio/Gemini, or LiteLLM direct mode.
    """

    def __init__(
        self,
        env: GymEnv | NotebookGymEnv,
        model: str | None = None,
        max_tokens: int = 8192,
        client: LLMClient | None = None,
    ):
        self.env = env
        self.model = model or default_model_name()
        self.max_tokens = max_tokens
        self.client = client or make_llm_client()
        self.messages: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def run(self) -> dict:
        context = self.env.reset()
        self.messages = [{"role": "user", "content": context["task"]}]
        max_agent_turns = max(self.env.state.max_steps * 2 + 5, 10)
        thoughts_on = bool(getattr(self.env, "enable_thoughts", False))
        # Goes through gym.prompts so a future dashboard-driven preset can
        # override blocks at run time without touching this call site.
        system_prompt = build_system_prompt(thoughts_on=thoughts_on)

        for turn in range(max_agent_turns):
            response = self.client.complete(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=self._messages_for_llm(),
            )
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
            self.messages.append({"role": "assistant", "content": response.text})

            try:
                action = Action.from_llm_response(response.text)
            except ActionParseError as exc:
                self._record_agent_trace(
                    {
                        "turn": turn + 1,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "raw_response": response.text,
                        "parse_status": "error",
                        "parse_error": str(exc),
                        "parsed_action": None,
                        "observation_action": None,
                        "done": False,
                        "submitted": False,
                    }
                )
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[ERROR] Could not parse your action: {exc}\n\n"
                            f"{ACTION_JSON_SCHEMA}"
                        ),
                    }
                )
                continue

            observation = self.env.step(action)
            self._record_agent_trace(
                {
                    "turn": turn + 1,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "raw_response": response.text,
                    "parse_status": "ok",
                    "parse_error": None,
                    "parsed_action": action.to_dict(),
                    "observation_action": observation.action,
                    "done": observation.done,
                    "submitted": observation.submitted,
                }
            )
            self.messages.append(
                {"role": "user", "content": self._build_feedback(observation)}
            )

            if observation.submitted:
                return self._build_summary()

            if observation.done:
                forced_observation = self._try_forced_submit()
                if forced_observation is not None:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": self._build_feedback(forced_observation),
                        }
                    )
                summary = self._build_summary()
                summary["forced_submit"] = True
                return summary

        summary = self._build_summary()
        summary["stopped_reason"] = "max_agent_turns"
        return summary

    def _build_summary(self) -> dict:
        summary = self.env.get_summary()
        summary["input_tokens"] = self.total_input_tokens
        summary["output_tokens"] = self.total_output_tokens
        summary["model"] = self.model
        return summary

    def _build_feedback(self, observation: Observation) -> str:
        feedback = observation.to_feedback_message()
        cell_history = getattr(self.env.state, "cell_history", None)
        if cell_history is None:
            return feedback
        notebook_context = cell_history.to_feedback_context(
            max_cells=3,
            max_code_chars=500,
            max_output_chars=250,
        )
        if notebook_context:
            feedback = f"{feedback}\n\n{notebook_context}"
        digest = getattr(self.env, "scratchpad_digest", None)
        if callable(digest):
            thoughts = digest()
            if thoughts:
                feedback = f"{feedback}\n\n{thoughts}"
        return feedback

    def _messages_for_llm(self) -> list[dict]:
        if os.getenv("AUTOVIBE_CONTEXT_COMPACTION", "off").lower() != "conservative":
            return self.messages
        try:
            last_turns = int(os.getenv("AUTOVIBE_CONTEXT_LAST_TURNS", "6"))
        except ValueError:
            last_turns = 6
        max_chars = int(os.getenv("AUTOVIBE_CONTEXT_MAX_CHARS", "12000"))
        context_pack = {}
        build_context_pack = getattr(self.env, "build_context_pack", None)
        if callable(build_context_pack):
            context_pack = build_context_pack()
        compact_message = {
            "role": "user",
            "content": "[CONTEXT PACK]\n" + json.dumps(context_pack, indent=2, ensure_ascii=False),
        }
        initial = self.messages[:1]
        recent = self.messages[-last_turns:] if last_turns > 0 else []
        packed = initial + [compact_message] + recent
        total = 0
        clipped: list[dict] = []
        for message in reversed(packed):
            content = str(message.get("content", ""))
            total += len(content)
            if total > max_chars and clipped:
                break
            clipped.append(message)
        return list(reversed(clipped))

    def _record_agent_trace(self, record: dict) -> None:
        recorder = getattr(self.env, "record_agent_turn", None)
        if callable(recorder):
            recorder(record)

    def _try_forced_submit(self) -> Observation | None:
        # Notebook environments expose a host-controlled finalize() that runs a
        # clean replay, validates a candidate variable, and submits it — so an
        # agent that built a good model but mismanaged the submit protocol still
        # yields a real score instead of null.
        finalize = getattr(self.env, "finalize", None)
        if callable(finalize):
            return finalize()

        workspace = getattr(self.env.state, "workspace", None)
        if workspace is not None:
            model_var, _ = workspace.first_existing(["best_model", "model"])
            if model_var is None:
                for key, value in workspace.namespace.items():
                    if not key.startswith("_") and callable(getattr(value, "predict", None)):
                        model_var = key
                        break
            if model_var is None:
                return None
            return self.env.step(Action.submit_action(model_var))

        candidates = getattr(self.env, "candidates", None)
        latest = candidates.latest() if candidates is not None else None
        if latest is not None:
            return self.env.step(Action.submit_action(latest.model_var))
        return None

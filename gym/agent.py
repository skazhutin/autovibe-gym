from __future__ import annotations

from .env import GymEnv
from .llm import LLMClient, default_model_name, make_llm_client
from .notebook_env import NotebookGymEnv
from .protocol import ACTION_JSON_SCHEMA, Action, ActionParseError, Observation


def _default_client() -> LLMClient:
    """Backward-compatible helper used by older experiment scripts."""
    return make_llm_client()


SYSTEM_PROMPT = f"""You are an expert data scientist solving a supervised machine learning task.

You work in a real Jupyter notebook controlled through JSON actions. Each turn
you must choose exactly one action.

Available kernel variables:
  train_df   - training DataFrame
  val_df     - validation DataFrame
  target_col - target column name (string)
  pd, np     - pandas and numpy

Installed ML libraries you may use:
  scikit-learn, xgboost, lightgbm, pandas, numpy.

CRITICAL RULES:
- Do not access test data; it is hidden until final submit.
- You can add, update, delete, move, inspect, and execute notebook cells.
- The kernel is persistent, but final acceptance depends on a clean
  restart_and_run_all of the current notebook.
- Before validate or submit, run restart_and_run_all successfully.
- Reserve your final steps for the acceptance lifecycle: restart_and_run_all,
  validate, then submit. A high validation printout is not a submission.
- The candidate must reproduce all preprocessing when predict() is called on
  raw validation or hidden-test rows.
- If you receive [MODEL CHECK] feedback, fix the candidate artifact before
  trying to submit.
- Use validation data for model selection.
- Assign your best trained candidate to a top-level variable called exactly
  `model`. If you wrap preprocessing, set `model = YourWrapper(...)`; do not
  leave the final artifact only in a local variable or under an arbitrary name.
- Submit only after validate succeeds on the same clean notebook revision.
- Return JSON only. Do not wrap it in markdown or add explanation.

{ACTION_JSON_SCHEMA}
"""


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

        for _ in range(max_agent_turns):
            response = self.client.complete(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=self.messages,
            )
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
            self.messages.append({"role": "assistant", "content": response.text})

            try:
                action = Action.from_llm_response(response.text)
            except ActionParseError as exc:
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
        return feedback

    def _try_forced_submit(self) -> Observation | None:
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

        if isinstance(self.env, NotebookGymEnv):
            return self._try_forced_notebook_submit()

        candidates = getattr(self.env, "candidates", None)
        latest = candidates.latest() if candidates is not None else None
        if latest is not None:
            return self.env.step(Action.submit_action(latest.model_var))
        return None

    def _try_forced_notebook_submit(self) -> Observation | None:
        """Finalize the current notebook without allowing more generated code.

        NotebookGymEnv intentionally requires clean replay and host validation
        before hidden-test submit. At budget exhaustion the agent cannot add or
        edit cells, but the host can still replay the notebook, validate an
        existing candidate, and submit it once. This turns "forgot to call
        submit" into an auditable finalization attempt instead of a silent null
        score, while preserving the hidden split gate.
        """
        env = self.env
        candidates = env.candidates
        latest = candidates.latest()
        if latest is not None:
            return env.submit_by_name(latest.model_var)

        if env.dirty_since_clean_run or env.last_clean_run_id is None:
            clean = env.restart_and_run_all()
            if clean.stderr.strip():
                return clean

        model_vars = self._notebook_candidate_names()
        last_observation: Observation | None = None
        for model_var in model_vars:
            validated = env.validate_candidate(model_var)
            last_observation = validated
            if validated.validation_metric is None or validated.stderr.strip():
                continue
            return env.submit_by_name(model_var)
        return last_observation

    def _notebook_candidate_names(self) -> list[str]:
        env = self.env
        discovered = env.candidate_variable_names()
        ordered: list[str] = []
        for name in ("best_model", "model", *discovered):
            if name not in ordered:
                ordered.append(name)
        return ordered

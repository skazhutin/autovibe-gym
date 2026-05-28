import os

from .env import GymEnv
from .llm import LLMClient, OpenAICompatibleLLMClient
from .protocol import ACTION_JSON_SCHEMA, Action, ActionParseError, Observation


SYSTEM_PROMPT = f"""You are an expert data scientist solving a supervised machine learning task.

You work in an iterative AutoML Gym. Each turn you must choose exactly one action.

Available workspace variables:
  train_df   - training DataFrame (may contain categorical columns)
  val_df     - validation DataFrame
  target_col - target column name (string)
  pd, np     - pandas and numpy

CRITICAL RULES:
- Do not access test data; it is hidden until submit.
- Treat each code action like a new notebook cell: reuse prior workspace
  variables instead of rewriting the whole solution from scratch.
- ALWAYS wrap your preprocessing + model in a sklearn Pipeline:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    model = Pipeline([('prep', preprocessor), ('clf', classifier)])
  This ensures model.predict() works correctly on raw unseen data at submit time.
  Never encode data manually with pd.get_dummies outside a Pipeline — the same
  transform must apply to both validation and test data automatically.
- If you receive [MODEL CHECK] feedback, fix the submitted model/pipeline before
  trying to submit.
- Use validation data for model selection.
- Assign your best trained pipeline/model to a variable called `model`.
- Submit only when your best model is ready and assigned.
- Return JSON only. Do not wrap it in markdown or add explanation.

{ACTION_JSON_SCHEMA}
"""


class GymAgent:
    """
    LLM agent that interacts with GymEnv through explicit JSON actions.

    The default client is OpenAI-compatible and is configured via:
      LLM_BASE_URL  - e.g. http://localhost:8000/v1
      LLM_API_KEY   - API key (use "local" for vLLM without auth)
      LLM_MODEL     - model name as served by the endpoint
    """

    def __init__(
        self,
        env: GymEnv,
        model: str | None = None,
        max_tokens: int = 8192,
        client: LLMClient | None = None,
    ):
        self.env = env
        self.model = model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.max_tokens = max_tokens
        self.client = client or OpenAICompatibleLLMClient()
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
        notebook_context = self.env.state.cell_history.to_feedback_context(
            max_cells=3,
            max_code_chars=500,
            max_output_chars=250,
        )
        if notebook_context:
            feedback = f"{feedback}\n\n{notebook_context}"
        return feedback

    def _try_forced_submit(self) -> Observation | None:
        model_var, _ = self.env.state.workspace.first_existing(["best_model", "model"])
        if model_var is None:
            return None
        return self.env.step(Action.submit_action(model_var))

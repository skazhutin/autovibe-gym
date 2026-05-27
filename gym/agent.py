import os
from openai import OpenAI

from .env import GymEnv, StepResult

SYSTEM_PROMPT = """You are an expert data scientist solving a supervised machine learning task.

You work in an iterative environment. Each turn you write Python code that gets executed.
Available variables in your namespace:
  train_df   — training DataFrame
  val_df     — validation DataFrame
  target_col — name of the target column (string)

Rules:
- Do NOT access test data — it is hidden until you call env.submit(model).
- Write clean, executable Python. Do not write markdown, only code blocks.
- After each step you may receive hints about things you might have missed. Take them seriously.
- When you have your best model ready, output exactly: SUBMIT
  (the runner will extract your best model from the namespace and call env.submit)
"""


def _build_feedback_message(result: StepResult, budget: int) -> str:
    parts = []
    if result.stdout.strip():
        parts.append(f"[OUTPUT]\n{result.stdout.strip()}")
    if result.stderr.strip():
        parts.append(f"[ERROR]\n{result.stderr.strip()}")
    if result.hints:
        hints_text = "\n".join(f"- {h}" for h in result.hints)
        parts.append(f"[HINTS]\n{hints_text}")
    parts.append(f"[BUDGET] {budget} steps remaining.")
    return "\n\n".join(parts)


def _make_client() -> OpenAI:
    """
    Build OpenAI-compatible client.
    Works with: vLLM (local), OpenAI, any OpenAI-compatible proxy.
    """
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("LLM_API_KEY", "local")
    return OpenAI(base_url=base_url, api_key=api_key)


class GymAgent:
    """
    LLM agent that interacts with GymEnv via any OpenAI-compatible API.
    Configure via environment variables:
      LLM_BASE_URL  — e.g. http://localhost:8000/v1  (vLLM) or https://api.openai.com/v1
      LLM_API_KEY   — API key (use "local" for vLLM without auth)
      LLM_MODEL     — model name as served by the endpoint
    """

    def __init__(
        self,
        env: GymEnv,
        model: str | None = None,
        max_tokens: int = 8192,
    ):
        self.env = env
        self.model = model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.max_tokens = max_tokens
        self.client = _make_client()
        self.messages: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def run(self) -> dict:
        context = self.env.reset()
        self.messages = [{"role": "user", "content": context["task"]}]

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.messages,
            )
            usage = response.usage
            if usage:
                self.total_input_tokens += usage.prompt_tokens
                self.total_output_tokens += usage.completion_tokens

            llm_text = response.choices[0].message.content.strip()
            self.messages.append({"role": "assistant", "content": llm_text})

            if "SUBMIT" in llm_text.upper():
                model_obj = (
                    self.env.state.namespace.get("model")
                    or self.env.state.namespace.get("best_model")
                )
                if model_obj is None:
                    feedback = (
                        "[ERROR] No variable named 'model' or 'best_model' found in namespace. "
                        "Train and assign your model first."
                    )
                    self.messages.append({"role": "user", "content": feedback})
                    continue
                self.env.submit(model_obj)
                return self._build_summary()

            code = self._extract_code(llm_text)
            result = self.env.step(code)
            feedback = _build_feedback_message(result, self.env.budget_remaining())
            self.messages.append({"role": "user", "content": feedback})

            if result.done:
                model_obj = (
                    self.env.state.namespace.get("model")
                    or self.env.state.namespace.get("best_model")
                )
                if model_obj is not None:
                    self.env.submit(model_obj)
                summary = self._build_summary()
                summary["forced_submit"] = True
                return summary

    def _build_summary(self) -> dict:
        summary = self.env.get_summary()
        summary["input_tokens"] = self.total_input_tokens
        summary["output_tokens"] = self.total_output_tokens
        summary["model"] = self.model
        return summary

    @staticmethod
    def _extract_code(text: str) -> str:
        if "```python" in text:
            start = text.index("```python") + len("```python")
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        return text

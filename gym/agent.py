from anthropic import Anthropic

from .env import GymEnv, StepResult


SYSTEM_PROMPT = """You are an expert data scientist solving a supervised machine learning task.

You work in an iterative environment. Each turn you write Python code that gets executed.
Available variables in your namespace:
  train_df   — training DataFrame
  val_df     — validation DataFrame
  target_col — name of the target column (string)

Rules:
- Do NOT access test data — it is hidden until you call env.submit(model).
- Write clean, executable Python. Do not write markdown, only code.
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


class GymAgent:
    """
    Single LLM agent that interacts with GymEnv via the Anthropic API.
    Runs until submit or budget exhausted.
    """

    def __init__(self, env: GymEnv, model: str = "claude-sonnet-4-6", max_tokens: int = 4096):
        self.env = env
        self.model = model
        self.max_tokens = max_tokens
        self.client = Anthropic()
        self.messages: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def run(self) -> dict:
        context = self.env.reset()
        self.messages = [{"role": "user", "content": context["task"]}]

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=self.messages,
            )
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens

            llm_text = response.content[0].text.strip()
            self.messages.append({"role": "assistant", "content": llm_text})

            if "SUBMIT" in llm_text.upper():
                model_obj = self.env.state.namespace.get("model") or self.env.state.namespace.get("best_model")
                if model_obj is None:
                    feedback = "[ERROR] No variable named 'model' or 'best_model' found in namespace. Train and assign your model first."
                    self.messages.append({"role": "user", "content": feedback})
                    continue
                score = self.env.submit(model_obj)
                summary = self.env.get_summary()
                summary["input_tokens"] = self.total_input_tokens
                summary["output_tokens"] = self.total_output_tokens
                return summary

            code = self._extract_code(llm_text)
            result = self.env.step(code)
            feedback = _build_feedback_message(result, self.env.budget_remaining())
            self.messages.append({"role": "user", "content": feedback})

            if result.done:
                # Budget exhausted — try to submit whatever is in namespace
                model_obj = self.env.state.namespace.get("model") or self.env.state.namespace.get("best_model")
                if model_obj is not None:
                    score = self.env.submit(model_obj)
                summary = self.env.get_summary()
                summary["input_tokens"] = self.total_input_tokens
                summary["output_tokens"] = self.total_output_tokens
                summary["forced_submit"] = True
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

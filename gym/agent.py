from __future__ import annotations

import json
import os

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
  scikit-learn, xgboost, lightgbm, catboost, pandas, numpy, matplotlib,
  seaborn, plotly, optuna, shap, imbalanced-learn (imblearn), category_encoders,
  statsmodels, tabulate.

CRITICAL RULES:
- Do not access test data; it is hidden until final submit.
- You can add, update, delete, move, inspect, and execute notebook cells.
- The kernel is persistent, but final acceptance depends on a clean
  restart_and_run_all of the current notebook.
- Before validate or submit, run restart_and_run_all successfully.
- The candidate must reproduce all preprocessing when predict() is called on
  raw validation or hidden-test rows.
- First create a simple robust candidate that can validate and submit on raw
  rows. Only after that, spend remaining steps on improvements. Never leave
  the episode without at least one raw-inference-ready candidate.
- If you create derived features from existing columns, the derivation must be
  inside the final model object. Do not train on X_train_processed and then
  submit a model that expects processed columns. Use sklearn Pipeline,
  ColumnTransformer, FunctionTransformer, or a custom sklearn-compatible
  transformer so model.predict(raw_df) works. This applies to date/time
  parsing, text/string features, target encoding, scaling, imputation, and
  column dropping.
- If you receive [MODEL CHECK] feedback, fix the candidate artifact before
  trying to submit.
- Use validation data for model selection.
- Assign your best trained candidate to a variable called `model`, or pass its
  variable name in validate/submit. Use model_var="auto" if unsure.
- Submit only after validate succeeds on the same clean notebook revision.
- Return JSON only. Do not wrap it in markdown or add explanation.

PREFER ENVIRONMENT TOOLS WHEN HELPFUL:
- inspect_data for compact EDA.
- profile_data for deeper data quality signals.
- check_candidate before finalizing.
- quick_validate for exploratory validation.
- list_candidates if unsure what model variables exist.
- cleanlab_diagnose for optional label-quality diagnostics in classification.
- tune_hyperparameters for bounded tuning of an existing candidate.
- finalize when ready or when budget is low.

AVOID COMMON FAILURE PATTERNS:
- fitting encoders/scalers outside the submitted model.
- creating train-only columns and submitting a model expecting those columns.
- using lambdas/local functions that cannot be serialized.
- relying on variables created only in the live kernel but not in the notebook.
- running large grid searches before a validated baseline exists.

FINALIZE EARLY — DO NOT RUN OUT OF STEPS:
- The single most important thing is to SUBMIT a validated candidate. As soon
  as you have a trained `model`, finalize immediately:
  restart_and_run_all -> validate/finalize -> submit. Budget a few steps for this.
- Do not spend steps polishing, re-running EDA, or deleting cells one by one.
  Prefer not to add throwaway cells rather than deleting them later.
- If you run out of steps, the environment will auto-finalize your latest
  model, but ONLY if a clean restart_and_run_all succeeds — so keep the
  notebook reproducible at all times.

IF SUBMIT FAILS ON HIDDEN DATA:
- If you get a "Submit failed on hidden test set" blocker you have retries.
  This means your model raised an error on private held-out rows. Fix it:
  1. Make the Pipeline robust: handle unseen categories (OrdinalEncoder with
     handle_unknown='use_encoded_value'), missing values, mixed dtypes.
  2. Never apply any transformation outside the Pipeline.
  3. Then: restart_and_run_all → validate → submit again.
- Do NOT skip the restart_and_run_all + validate cycle before resubmitting.

KEEP THE CLEAN RUN FAST AND ROBUST:
- restart_and_run_all re-executes EVERY cell top-to-bottom on a fresh kernel,
  and each cell must finish within the cell time limit. One failing or slow
  cell aborts the whole clean run, so no candidate is accepted.
- Only import libraries you actually use; a single failed import anywhere
  aborts the entire clean run.
- Keep hyperparameter search small enough to finish well within the time
  limit: prefer tune_hyperparameters, a tiny grid, or RandomizedSearchCV with
  few iterations and cv<=3. Use n_jobs=-1 where safe.

{ACTION_JSON_SCHEMA}
"""

THOUGHTS_ENABLED_PROMPT = """

Thoughts mode is enabled.
Every JSON action must include `thoughts`.
Use `thoughts` for a short visible summary of what you learned, what you are doing, why you are doing it, or what you plan to do next.
Your first action must be:
{"type": "think", "stage": "planning", "thoughts": "..."}.
"""

THOUGHTS_DISABLED_PROMPT = """

Thoughts mode is disabled.
Do not include `thoughts`.
Do not use type `think`.
Do not use stage `planning`.
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
        thoughts_on = bool(getattr(self.env, "enable_thoughts", False))
        system_prompt = SYSTEM_PROMPT + (
            THOUGHTS_ENABLED_PROMPT if thoughts_on else THOUGHTS_DISABLED_PROMPT
        )

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

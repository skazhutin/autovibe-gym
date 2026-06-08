"""Block-structured system prompt for the gym/iterative agent.

The historical monolithic ``SYSTEM_PROMPT`` constant in ``gym.agent`` is now
assembled from named blocks defined here, plus the canonical
``ACTION_JSON_SCHEMA`` from ``gym.protocol``. This split exists so the
dashboard can expose a /prompts editor where each block has its own tier
(locked / trusted / editable) without giving the operator a single giant
textarea where one accidental deletion silently breaks the agent↔env
contract.

INVARIANT: ``build_system_prompt()`` with no overrides + the default
thoughts toggles must produce **byte-identical** output to the historical
``SYSTEM_PROMPT + THOUGHTS_*_PROMPT`` concatenation. Enforced by
``tests/test_prompts.py`` against pinned SHA256 baselines. Do not change
DEFAULT_BLOCKS, DEFAULT_THOUGHTS_*, or the join layout without updating the
baselines in that test (and accept that this shifts the experiment baseline
for the whole project).
"""
from __future__ import annotations

from .protocol import ACTION_JSON_SCHEMA


# Order matters: blocks are joined with "\n\n" in this sequence.
BLOCK_ORDER: list[str] = [
    "header",
    "kernel_vars",
    "libraries",
    "critical_rules",
    "tools_hint",
    "failure_patterns",
    "finalize",
    "submit_recovery",
    "clean_run_tips",
]


# Tier policy for the dashboard editor:
#   locked   — read-only; tied to code (kernel variable names, JSON schema).
#              An operator override would diverge from the runtime contract.
#   trusted  — editable, but the editor confirms before first change: removing
#              a sentence here is a common way to silently break the agent.
#   editable — free to edit; these blocks are tone/strategy hints whose loss
#              degrades quality but does not break the protocol.
# Two synthetic tiers cover the thoughts-toggle suffixes (not in BLOCK_ORDER).
BLOCK_TIERS: dict[str, str] = {
    "header": "trusted",
    "kernel_vars": "locked",
    "libraries": "trusted",
    "critical_rules": "trusted",
    "tools_hint": "editable",
    "failure_patterns": "editable",
    "finalize": "editable",
    "submit_recovery": "trusted",
    "clean_run_tips": "editable",
    "thoughts_on": "trusted",
    "thoughts_off": "trusted",
}


# Each block stores its body without a trailing newline; join("\n\n") restores
# the blank-line separators of the original prompt. Do not add or remove blank
# lines inside the strings — the byte-identity test will catch it.
DEFAULT_BLOCKS: dict[str, str] = {
    "header": (
        "You are an expert data scientist solving a supervised machine learning task.\n"
        "\n"
        "You work in a real Jupyter notebook controlled through JSON actions. Each turn\n"
        "you must choose exactly one action."
    ),
    "kernel_vars": (
        "Available kernel variables:\n"
        "  train_df   - training DataFrame\n"
        "  val_df     - validation DataFrame\n"
        "  target_col - target column name (string)\n"
        "  pd, np     - pandas and numpy"
    ),
    "libraries": (
        "Installed ML libraries you may use:\n"
        "  scikit-learn, xgboost, lightgbm, catboost, pandas, numpy, matplotlib,\n"
        "  seaborn, plotly, optuna, shap, imbalanced-learn (imblearn), category_encoders,\n"
        "  statsmodels, tabulate."
    ),
    "critical_rules": (
        "CRITICAL RULES:\n"
        "- Do not access test data; it is hidden until final submit.\n"
        "- You can add, update, delete, move, inspect, and execute notebook cells.\n"
        "- The kernel is persistent, but final acceptance depends on a clean\n"
        "  restart_and_run_all of the current notebook.\n"
        "- Before validate or submit, run restart_and_run_all successfully.\n"
        "- The candidate must reproduce all preprocessing when predict() is called on\n"
        "  raw validation or hidden-test rows.\n"
        "- First create a simple robust candidate that can validate and submit on raw\n"
        "  rows. Only after that, spend remaining steps on improvements. Never leave\n"
        "  the episode without at least one raw-inference-ready candidate.\n"
        "- If you create derived features from existing columns, the derivation must be\n"
        "  inside the final model object. Do not train on X_train_processed and then\n"
        "  submit a model that expects processed columns. Use sklearn Pipeline,\n"
        "  ColumnTransformer, FunctionTransformer, or a custom sklearn-compatible\n"
        "  transformer so model.predict(raw_df) works. This applies to date/time\n"
        "  parsing, text/string features, target encoding, scaling, imputation, and\n"
        "  column dropping.\n"
        "- If you receive [MODEL CHECK] feedback, fix the candidate artifact before\n"
        "  trying to submit.\n"
        "- Use validation data for model selection.\n"
        "- Assign your best trained candidate to a variable called `model`, or pass its\n"
        "  variable name in validate/submit. Use model_var=\"auto\" if unsure.\n"
        "- Submit only after validate succeeds on the same clean notebook revision.\n"
        "- Return JSON only. Do not wrap it in markdown or add explanation."
    ),
    "tools_hint": (
        "PREFER ENVIRONMENT TOOLS WHEN HELPFUL:\n"
        "- inspect_data for compact EDA.\n"
        "- profile_data for deeper data quality signals.\n"
        "- check_candidate before finalizing.\n"
        "- quick_validate for exploratory validation.\n"
        "- list_candidates if unsure what model variables exist.\n"
        "- cleanlab_diagnose for optional label-quality diagnostics in classification.\n"
        "- tune_hyperparameters for bounded tuning of an existing candidate.\n"
        "- finalize when ready or when budget is low."
    ),
    "failure_patterns": (
        "AVOID COMMON FAILURE PATTERNS:\n"
        "- fitting encoders/scalers outside the submitted model.\n"
        "- creating train-only columns and submitting a model expecting those columns.\n"
        "- using lambdas/local functions that cannot be serialized.\n"
        "- relying on variables created only in the live kernel but not in the notebook.\n"
        "- running large grid searches before a validated baseline exists."
    ),
    "finalize": (
        "FINALIZE EARLY — DO NOT RUN OUT OF STEPS:\n"
        "- The single most important thing is to SUBMIT a validated candidate. As soon\n"
        "  as you have a trained `model`, finalize immediately:\n"
        "  restart_and_run_all -> validate/finalize -> submit. Budget a few steps for this.\n"
        "- Do not spend steps polishing, re-running EDA, or deleting cells one by one.\n"
        "  Prefer not to add throwaway cells rather than deleting them later.\n"
        "- If you run out of steps, the environment will auto-finalize your latest\n"
        "  model, but ONLY if a clean restart_and_run_all succeeds — so keep the\n"
        "  notebook reproducible at all times."
    ),
    "submit_recovery": (
        "IF SUBMIT FAILS ON HIDDEN DATA:\n"
        "- If you get a \"Submit failed on hidden test set\" blocker you have retries.\n"
        "  This means your model raised an error on private held-out rows. Fix it:\n"
        "  1. Make the Pipeline robust: handle unseen categories (OrdinalEncoder with\n"
        "     handle_unknown='use_encoded_value'), missing values, mixed dtypes.\n"
        "  2. Never apply any transformation outside the Pipeline.\n"
        "  3. Then: restart_and_run_all → validate → submit again.\n"
        "- Do NOT skip the restart_and_run_all + validate cycle before resubmitting."
    ),
    "clean_run_tips": (
        "KEEP THE CLEAN RUN FAST AND ROBUST:\n"
        "- restart_and_run_all re-executes EVERY cell top-to-bottom on a fresh kernel,\n"
        "  and each cell must finish within the cell time limit. One failing or slow\n"
        "  cell aborts the whole clean run, so no candidate is accepted.\n"
        "- Only import libraries you actually use; a single failed import anywhere\n"
        "  aborts the entire clean run.\n"
        "- Keep hyperparameter search small enough to finish well within the time\n"
        "  limit: prefer tune_hyperparameters, a tiny grid, or RandomizedSearchCV with\n"
        "  few iterations and cv<=3. Use n_jobs=-1 where safe."
    ),
}


# Thoughts-mode suffixes: appended AFTER the assembled body + JSON schema.
# Both begin with "\n\n" to reproduce the historical f-string layout where the
# closing triple-quote contributes one newline and the THOUGHTS string adds
# its own leading blank line. Adjusting the leading/trailing whitespace here
# will shift the byte-identity baseline.
DEFAULT_THOUGHTS_ON: str = (
    "\n"
    "\n"
    "Thoughts mode is enabled.\n"
    "Every JSON action must include `thoughts`.\n"
    "Use `thoughts` for a short visible summary of what you learned, what you are doing, why you are doing it, or what you plan to do next.\n"
    "Your first action must be:\n"
    "{\"type\": \"think\", \"stage\": \"planning\", \"thoughts\": \"...\"}.\n"
)

DEFAULT_THOUGHTS_OFF: str = (
    "\n"
    "\n"
    "Thoughts mode is disabled.\n"
    "Do not include `thoughts`.\n"
    "Do not use type `think`.\n"
    "Do not use stage `planning`.\n"
)


# Block names whose override the public API silently ignores. These are
# reserved for code-driven content (kernel variable names, JSON schema). The
# JSON schema is not in BLOCK_ORDER at all — it is interpolated at assembly
# time from gym.protocol.ACTION_JSON_SCHEMA, the same source the runtime
# parser uses. Overriding it from a preset would diverge prompt from parser.
LOCKED_BLOCKS: frozenset[str] = frozenset(
    name for name, tier in BLOCK_TIERS.items() if tier == "locked"
)


def assemble_body(blocks: dict[str, str] | None = None) -> str:
    """Join blocks in canonical order + the JSON schema.

    Missing keys in ``blocks`` fall back to DEFAULT_BLOCKS. Locked block
    overrides are silently dropped — the dashboard layer is expected to reject
    them with a 422 before reaching here, but we double-guard so a hand-edited
    preset file cannot diverge the prompt from the runtime parser.
    """
    merged = dict(DEFAULT_BLOCKS)
    if blocks:
        for name, value in blocks.items():
            if name not in BLOCK_ORDER:
                continue
            if name in LOCKED_BLOCKS:
                continue
            merged[name] = value
    body = "\n\n".join(merged[name] for name in BLOCK_ORDER)
    return body + "\n\n" + ACTION_JSON_SCHEMA + "\n"


def build_system_prompt(
    blocks: dict[str, str] | None = None,
    *,
    thoughts_on: bool,
    thoughts_on_text: str | None = None,
    thoughts_off_text: str | None = None,
) -> str:
    """Assemble the full system prompt for one episode.

    ``blocks`` is a partial override map; missing keys use defaults. The
    ``thoughts_on_text`` / ``thoughts_off_text`` arguments let callers override
    the toggle suffix (also editable by the dashboard), but default to the
    historical strings. With all defaults, output is byte-identical to the
    legacy ``SYSTEM_PROMPT + THOUGHTS_*_PROMPT`` concatenation — guarded by
    ``tests/test_prompts.py``.
    """
    body = assemble_body(blocks)
    if thoughts_on:
        suffix = thoughts_on_text if thoughts_on_text is not None else DEFAULT_THOUGHTS_ON
    else:
        suffix = thoughts_off_text if thoughts_off_text is not None else DEFAULT_THOUGHTS_OFF
    return body + suffix


__all__ = [
    "BLOCK_ORDER",
    "BLOCK_TIERS",
    "DEFAULT_BLOCKS",
    "DEFAULT_THOUGHTS_ON",
    "DEFAULT_THOUGHTS_OFF",
    "LOCKED_BLOCKS",
    "assemble_body",
    "build_system_prompt",
]

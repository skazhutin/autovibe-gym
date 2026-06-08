"""Post-submit solution summary.

Once the model has SOLVED the task (made its final submit), we make ONE extra
LLM call that asks it to look back at the solution it just submitted and write a
retrospective summary of it. The result is saved as ``run_summary.json`` in the
episode workspace and rendered at the top of the dashboard «Мысли» tab.

This is a *retrospective* (past tense, "what I built and submitted"), not a
plan — the call happens after the final submit and is fed the final submitted
solution code so it describes the real solution, not the initial intentions.

Privacy: the summary is built only from the conversation/solution the model
already produced. The hidden test score is never part of that context, so asking
the model to summarize cannot leak it.

The call is best-effort: any client/provider error (rate limit, network, etc.)
is swallowed and the run proceeds without a summary rather than failing.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

SUMMARY_FILENAME = "run_summary.json"
FINAL_SOLUTION_FILENAME = "final_notebook.py"

SUMMARY_SYSTEM_PROMPT = """You have FINISHED the machine-learning task and
ALREADY SUBMITTED your final solution. Write a short retrospective summary of
that submitted solution, in English, for your teammates.

This text is shown directly in a dashboard UI. Do NOT output analysis,
reasoning, self-talk, confidence checks, translation notes, planning notes,
word-count checks, or any "draft/final polish" scaffolding.

Write in the PAST tense. This is a report on finished work, NOT a plan.

Start the very first character of your answer with `**What was built**`.
Return exactly four short markdown lines in this exact format:
**What was built** — ...
**How it was solved** — ...
**Result** — ...
**What to improve** — ...

Rules:
- Describe ONLY the solution you actually submitted.
- Do not restate your initial plan.
- Do not invent numbers, steps, or tools you never used.
- Maximum ~180 words total.
- No code blocks.
- No bullets or numbered lists.
- No headings other than the four bold section labels above.
- Do not write anything before the first section or after the fourth section.
- Output only the final summary text."""

# Used when no explicit solution code is supplied (the conversation already
# contains the submitted solution).
SUMMARY_REQUEST = (
    "You just submitted the final solution. Fill the four-line markdown "
    "template from the system prompt in English and in the past tense. "
    "Return only those four final lines, with no analysis, drafts, "
    "checklists, confidence scores, or notes about how you are writing it."
)

# Keep the post-run call cheap and within tight per-minute token budgets.
DEFAULT_MAX_TOKENS = 400
# How many trailing conversation messages to keep (plus the first task message)
# so the summary stays grounded in the final solution without blowing context.
DEFAULT_MAX_MESSAGES = 16
# Cap embedded solution code so a long notebook export stays within token limits.
_MAX_CODE_CHARS = 4000
_MAX_SUMMARY_WORDS = 180
_SECTION_WORD_CAPS = {
    "What was built": 36,
    "How it was solved": 60,
    "Result": 26,
    "What to improve": 34,
}
_SECTION_ORDER = (
    "What was built",
    "How it was solved",
    "Result",
    "What to improve",
)
_SECTION_ALIASES = {
    "What was built": ("what was built", "что построено", "подход"),
    "How it was solved": ("how it was solved", "как решал", "key steps", "ключевые шаги"),
    "Result": ("result", "результат"),
    "What to improve": ("what to improve", "что улучшить"),
}
_META_MARKERS = (
    "the user wants",
    "summary must follow",
    "constraint checklist",
    "confidence score",
    "drafting the content",
    "mental draft",
    "translating to russian",
    "word count check",
    "final polish",
    "analyze the request",
    "analyze the submitted solution",
    "review the history",
    "refine and translate",
    "format constraints",
    "labels:",
    "language:",
    "tense:",
    "the prompt asks",
)
_ESTIMATOR_NAMES = (
    "LGBMClassifier",
    "LGBMRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "LogisticRegression",
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SVC",
    "SVR",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
)
_PREPROCESSOR_NAMES = (
    "ColumnTransformer",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "OneHotEncoder",
    "OrdinalEncoder",
    "SimpleImputer",
    "KNNImputer",
    "PolynomialFeatures",
)
_PRETTY_NAME_ALIASES = {
    name: name for name in (*_ESTIMATOR_NAMES, *_PREPROCESSOR_NAMES)
}


def _trim_conversation(
    conversation: list[dict[str, Any]], max_messages: int
) -> list[dict[str, Any]]:
    """Keep the opening task message plus the most recent exchanges."""
    if max_messages <= 0 or len(conversation) <= max_messages:
        return list(conversation)
    head = conversation[:1]
    tail = conversation[-(max_messages - len(head)):]
    return head + tail


def _heading_candidate(line: str) -> str:
    text = line.strip()
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^\s*(?:[-*+]\s*)?(?:\d+[.)]\s*)?", "", text)
    return text.strip()


def _match_section(line: str) -> tuple[str | None, str]:
    candidate = _heading_candidate(line)
    lowered = candidate.casefold()
    for canonical in _SECTION_ORDER:
        for alias in _SECTION_ALIASES[canonical]:
            if lowered.startswith(alias):
                rest = candidate[len(alias):].lstrip(" \t:;,.!-—–*")
                return canonical, rest.strip()
    return None, ""


def _looks_like_meta_line(line: str) -> bool:
    lowered = _heading_candidate(line).casefold()
    if any(marker in lowered for marker in _META_MARKERS):
        return True
    return bool(
        re.match(
            r"^\d+[.)]?\s*(past tense|russian|specific sections|compact markdown|"
            r"describe only|no code blocks|no json|max\b|goal|context|structure|constraints)\b",
            lowered,
        )
    )


def _is_terminal_meta_line(line: str) -> bool:
    lowered = _heading_candidate(line).casefold()
    return any(marker in lowered for marker in ("format constraints", "labels:", "language:", "tense:"))


def _clean_body_lines(lines: list[str]) -> str:
    parts: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        line = re.split(
            r"\*?\*?(?:Format constraints|Constraint checklist|Analyze the submitted solution|The prompt asks)\b",
            line,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        if not line:
            continue
        if _looks_like_meta_line(line):
            continue
        line = re.sub(r"^\s*(?:[-*+]\s*)?(?:\d+[.)]\s*)?", "", line).strip()
        line = re.sub(r"(?<!\w)\d+[.)]\s+", "", line)
        parts.append(line)
    return re.sub(r"\s+", " ", " ".join(parts)).strip(" -–—")


def _word_count(text: str) -> int:
    return len(text.split())


def _limit_words(text: str, max_words: int = _MAX_SUMMARY_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    clipped = " ".join(words[:max_words]).rstrip(",;:-–—")
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        size = _word_count(sentence)
        if kept and count + size > max_words:
            break
        kept.append(sentence)
        count += size
        if count >= max_words:
            break
    if kept and count <= max_words:
        return " ".join(kept).strip()
    if clipped and clipped[-1] not in ".!?":
        clipped += "."
    return clipped


def normalize_summary_text(text: str) -> str:
    """Best-effort cleanup for verbose model outputs.

    Some models ignore the instruction and emit reasoning, drafting notes, or a
    full self-check before the real summary. We salvage the final four sections
    when possible and otherwise return a cleaned raw fallback.
    """
    raw = (text or "").replace("\r\n", "\n").strip()
    if not raw:
        return ""
    raw = re.sub(r"```(?:markdown|md|text)?\n?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "").strip()
    raw = re.sub(
        r"\*\*((?:What was built|How it was solved|Result|What to improve|"
        r"Что построено|Как решал|Результат|Что улучшить))\s*:?\*\*:?",
        r"\n**\1**",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(
        r"\s*-\s*\*\*((?:What was built|How it was solved|Result|What to improve|"
        r"Что построено|Как решал|Результат|Что улучшить))\s*:?\*\*:?",
        r"\n**\1**",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(
        r"\s*\*\*((?:What was built|How it was solved|Result|What to improve|"
        r"Что построено|Как решал|Результат|Что улучшить))\s*:?\*\*:?",
        r"\n**\1**",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(
        r"\s*(?:\*\*(?:Analyze the submitted solution|Format constraints)\s*:?\*\*:?"
        r"|Format constraints:|Constraint checklist.*?:|Analyze the submitted solution:|"
        r"What was built:|How it was solved:|Result:|What to improve:|"
        r"Что построено:|Как решал:|Результат:|Что улучшить:)",
        lambda m: "\n" + m.group(0).strip(),
        raw,
        flags=re.IGNORECASE,
    ).strip()

    sections: dict[str, list[str]] = {name: [] for name in _SECTION_ORDER}
    current: str | None = None
    for line in raw.split("\n"):
        if _is_terminal_meta_line(line):
            break
        section, rest = _match_section(line)
        if section:
            current = section
            sections[current] = [rest] if rest else []
            continue
        if _looks_like_meta_line(line):
            continue
        if current is not None:
            sections[current].append(line)

    built_sections: list[tuple[str, str]] = []
    raw_sections: list[tuple[str, str]] = []
    for name in _SECTION_ORDER:
        body = _clean_body_lines(sections[name])
        if body:
            raw_sections.append((name, body))
    total_words = sum(_word_count(body) for _, body in raw_sections)
    for name, body in raw_sections:
        if total_words > _MAX_SUMMARY_WORDS:
            body = _limit_words(body, _SECTION_WORD_CAPS.get(name, _MAX_SUMMARY_WORDS))
        built_sections.append((name, body))
    if len(built_sections) >= 2:
        return _limit_words(
            "\n".join(f"**{name}** — {body}" for name, body in built_sections)
        )

    fallback_lines = [
        line.strip()
        for line in raw.split("\n")
        if line.strip() and not _looks_like_meta_line(line)
    ]
    return _limit_words("\n".join(fallback_lines).strip())


def _parsed_summary_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for raw in (text or "").replace("\r\n", "\n").split("\n"):
        section, rest = _match_section(raw)
        if section and rest.strip():
            sections.append((section, rest.strip()))
    return sections


def summary_needs_fallback(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    lowered = cleaned.casefold()
    if any(marker in lowered for marker in _META_MARKERS):
        return True
    sections = _parsed_summary_sections(cleaned)
    if len(sections) < 3:
        return True
    for _, body in sections:
        if any(marker in body.casefold() for marker in _META_MARKERS):
            return True
        if _word_count(body) < 3:
            return True
        if len(re.sub(r"[^A-Za-z0-9]+", "", body)) < 8:
            return True
    return False


def _pretty_name(name: str) -> str:
    return _PRETTY_NAME_ALIASES.get(
        name,
        re.sub(r"(?<!^)([A-Z])", r" \1", name)
        .replace("X G B", "XGB")
        .replace("L G B M", "LGBM"),
    )


def _join_natural(parts: list[str]) -> str:
    items = [part for part in parts if part]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def fallback_summary_from_solution(
    solution_code: str | None,
    *,
    validation_metric: float | None = None,
) -> str:
    """Heuristic English fallback when the model output is unusable.

    This keeps the dashboard readable for runs where the summarizer emits
    planning/reasoning text instead of the requested four-line retrospective.
    """
    code = (solution_code or "").strip()
    if not code:
        return ""

    estimator = next(
        (
            name
            for name in _ESTIMATOR_NAMES
            if re.search(rf"\b{name}\s*\(", code)
        ),
        None,
    )
    preprocessors = [
        _pretty_name(name)
        for name in _PREPROCESSOR_NAMES
        if name != "ColumnTransformer" and re.search(rf"\b{name}\s*\(", code)
    ]
    uses_pipeline = bool(re.search(r"\bPipeline\s*\(", code))
    uses_column_transformer = bool(re.search(r"\bColumnTransformer\s*\(", code))
    uses_select_dtypes = "select_dtypes" in code
    uses_raw_val = "val_df" in code
    uses_target_split = "target_col" in code and ".drop(columns=[" in code
    uses_train_split = "train_df" in code

    estimator_label = _pretty_name(estimator) if estimator else "tabular model"
    prep_label = _join_natural(preprocessors[:3])

    if uses_pipeline:
        what = f"Built a scikit-learn Pipeline around {estimator_label}."
    else:
        what = f"Built a {estimator_label} solution."
    if prep_label:
        what = what[:-1] + f" with {prep_label}."
    elif uses_column_transformer:
        what = what[:-1] + " with a ColumnTransformer-based preprocessing step."

    how_parts: list[str] = []
    if uses_target_split:
        how_parts.append("Separated features from the target with `target_col`.")
    if uses_select_dtypes:
        how_parts.append("Selected numeric or categorical feature groups directly from the raw DataFrame.")
    elif "X_train.columns.tolist()" in code or "numeric_features = X_train.columns" in code:
        how_parts.append("Used the available feature columns directly without a separate feature generation stage.")
    if prep_label and uses_column_transformer:
        how_parts.append(f"Applied {prep_label} inside a ColumnTransformer.")
    elif prep_label:
        how_parts.append(f"Applied {prep_label} before fitting the estimator.")
    if uses_train_split and uses_raw_val:
        how_parts.append("Trained on `train_df` and validated on `val_df`.")
    elif "train_test_split" in code:
        how_parts.append("Used an explicit train/validation split before the final submission.")
    how = " ".join(how_parts) or "Fit the submitted model directly on the prepared training data and kept the training flow simple."

    result = (
        f"Best validation metric was {validation_metric:.4f}."
        if validation_metric is not None
        else "Validated the submitted model before the final submission."
    )

    improve_parts: list[str] = []
    if not re.search(r"\b(GridSearchCV|RandomizedSearchCV|Optuna|optuna|study\.optimize)\b", code):
        improve_parts.append("tune the main hyperparameters")
    if estimator not in {"LGBMClassifier", "LGBMRegressor", "XGBClassifier", "XGBRegressor", "CatBoostClassifier", "CatBoostRegressor", "HistGradientBoostingClassifier", "HistGradientBoostingRegressor"}:
        improve_parts.append("compare against stronger boosting baselines")
    if "PolynomialFeatures" not in code and "feature_engine" not in code.lower():
        improve_parts.append("add targeted feature engineering only if the dataset shows a clear gap")
    improve = "Next step would be to " + _join_natural(improve_parts[:3]) + "."

    return "\n".join(
        [
            f"**What was built** — {what}",
            f"**How it was solved** — {how}",
            f"**Result** — {result}",
            f"**What to improve** — {improve}",
        ]
    )


def _build_request(solution_code: str | None) -> str:
    """The final user turn that asks for the retrospective summary, optionally
    embedding the exact solution code that was just submitted."""
    code = (solution_code or "").strip()
    if not code:
        return SUMMARY_REQUEST
    if len(code) > _MAX_CODE_CHARS:
        code = code[:_MAX_CODE_CHARS] + "\n# … (truncated)"
    return (
        "You just submitted the final solution. Below is the exact code of the "
        "submitted solution. Write a retrospective summary of that solution in "
        "English and in the past tense, using exactly the structure from the "
        "system prompt (What was built / How it was solved / Result / What to "
        "improve). Do not show analysis, drafts, checklists, confidence "
        "scores, word-count checks, or any notes about how you are writing the "
        "answer. Return only the final markdown summary text.\n\n"
        "=== Final submitted solution ===\n"
        f"{code}"
    )


def read_solution_code(workspace_dir: str | Path | None) -> str | None:
    """Best-effort load of the exported final notebook code for notebook modes."""
    if not workspace_dir:
        return None
    path = Path(workspace_dir) / FINAL_SOLUTION_FILENAME
    try:
        text = path.read_text("utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return text or None


def generate_summary(
    client: Any,
    model: str,
    *,
    conversation: list[dict[str, Any]] | None = None,
    solution_code: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_messages: int = DEFAULT_MAX_MESSAGES,
) -> dict[str, Any] | None:
    """Make one best-effort LLM call summarizing the just-submitted solution.

    ``conversation`` is the message list the model worked with (task prompt, its
    own actions, environment feedback). ``solution_code``, when given, is the
    final submitted solution and is embedded into the request so the summary
    describes the real solution. Returns a JSON-serializable payload, or ``None``
    if the call failed or produced nothing.
    """
    if client is None or not model:
        return None
    messages = _trim_conversation(list(conversation or []), max_messages)
    messages.append({"role": "user", "content": _build_request(solution_code)})
    try:
        response = client.complete(
            model=model,
            max_tokens=min(DEFAULT_MAX_TOKENS, max(64, int(max_tokens))),
            system=SUMMARY_SYSTEM_PROMPT,
            messages=messages,
        )
    except Exception:
        # Best-effort: a summary must never break or fail a finished run.
        return None
    text = normalize_summary_text((getattr(response, "text", "") or "").strip())
    if summary_needs_fallback(text):
        fallback = fallback_summary_from_solution(solution_code)
        if fallback:
            text = fallback
    if not text:
        return None
    return {
        "summary": text,
        "model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_tokens": int(getattr(response, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(response, "output_tokens", 0) or 0),
    }


def write_summary(workspace_dir: str | Path | None, payload: dict[str, Any] | None) -> None:
    """Persist the summary payload as run_summary.json in the episode workspace."""
    if not payload or not workspace_dir:
        return
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / SUMMARY_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), "utf-8"
    )


def generate_and_write(
    client: Any,
    model: str,
    workspace_dir: str | Path | None,
    *,
    conversation: list[dict[str, Any]] | None = None,
    solution_code: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_messages: int = DEFAULT_MAX_MESSAGES,
) -> dict[str, Any] | None:
    """Convenience: generate the summary and write it to the workspace."""
    payload = generate_summary(
        client,
        model,
        conversation=conversation,
        solution_code=solution_code,
        max_tokens=max_tokens,
        max_messages=max_messages,
    )
    write_summary(workspace_dir, payload)
    return payload

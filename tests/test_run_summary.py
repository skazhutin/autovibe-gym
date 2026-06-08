import json

import pytest

from gym import run_summary
from gym.llm import LLMResponse


class FakeClient:
    def __init__(
        self,
        text=(
            "**What was built** — forest.\n"
            "**How it was solved** — trained a baseline model.\n"
            "**Result** — val = 0.7.\n"
            "**What to improve** — tune hyperparameters."
        ),
        *,
        raises=False,
    ):
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

    assert out["summary"].startswith("**What was built**")
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


def test_generate_summary_embeds_final_solution_code_when_provided():
    client = FakeClient()
    convo = [{"role": "user", "content": "task"}]

    run_summary.generate_summary(
        client,
        "m",
        conversation=convo,
        solution_code="model = pipeline.fit(train_df, y)",
    )

    sent = client.calls[0]["messages"]
    assert "Final submitted solution" in sent[-1]["content"]
    assert "model = pipeline.fit(train_df, y)" in sent[-1]["content"]
    assert sent[0] == convo[0]


def test_generate_summary_strips_reasoning_and_keeps_only_final_sections():
    client = FakeClient(
        text=(
            "The user wants a retrospective summary of the submitted solution in Russian.\n\n"
            "Constraint checklist & Confidence score:\n"
            "1. Past tense? Yes.\n\n"
            "Drafting the content:\n"
            "What was built: Draft.\n"
            "How it was solved: Draft.\n"
            "Result: Draft.\n"
            "What to improve: Draft.\n\n"
            "Final Polish (Markdown):\n"
            "What was built: Built a sklearn Pipeline with LightGBM.\n"
            "How it was solved: Used numeric features only; trained on train_df and checked on val_df.\n"
            "Result: Achieved F1 Macro = 0.9411 on validation.\n"
            "What to improve: Tune hyperparameters and handle class imbalance.\n"
        )
    )

    out = run_summary.generate_summary(client, "m", conversation=[])

    assert out is not None
    assert out["summary"].startswith("**What was built** — Built a sklearn Pipeline with LightGBM.")
    assert "**How it was solved** — Used numeric features only; trained on train_df and checked on val_df." in out["summary"]
    assert "Constraint checklist" not in out["summary"]
    assert "Drafting the content" not in out["summary"]
    assert "Final Polish" not in out["summary"]


def test_generate_summary_falls_back_to_solution_code_when_output_is_unusable():
    client = FakeClient(
        text=(
            "* **Context:** The user has already submitted a solution.\n"
            "* **Constraints:** no bullets, no analysis.\n"
            "3. **Drafting the Sections:**\n"
            "**What was built** I built a model.\n"
        )
    )

    out = run_summary.generate_summary(
        client,
        "m",
        conversation=[],
        solution_code=(
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "model = Pipeline(steps=[('scale', StandardScaler()), ('clf', RandomForestClassifier())])\n"
            "X_train = train_df.drop(columns=[target_col])\n"
            "model.fit(X_train, train_df[target_col])\n"
        ),
    )

    assert out is not None
    assert out["summary"].startswith("**What was built** — Built a scikit-learn Pipeline around RandomForestClassifier")
    assert "**Result** —" in out["summary"]
    assert "Constraints" not in out["summary"]


def test_generate_summary_falls_back_when_sections_are_placeholder_garbage():
    client = FakeClient(
        text=(
            "**What was built** — `, `\n"
            "**How it was solved** — `, `\n"
            "**Result** — `, `\n"
            "**What to improve** — `."
        )
    )

    out = run_summary.generate_summary(
        client,
        "m",
        conversation=[],
        solution_code=(
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "X_train = train_df.drop(columns=[target_col])\n"
            "model = Pipeline(steps=[('scale', StandardScaler()), ('clf', RandomForestClassifier())])\n"
        ),
    )

    assert out is not None
    assert out["summary"].startswith("**What was built** — Built a scikit-learn Pipeline around RandomForestClassifier")


def test_normalize_summary_handles_single_line_meta_and_inline_sections():
    raw = (
        "The solution is a baseline RandomForest pipeline. **Analyze the submitted solution:** "
        "- **What was built:** A sklearn Pipeline with StandardScaler and RandomForestClassifier. "
        "- **How it was solved:** Inspected data, trained on train_df, validated on val_df. "
        "- **Result:** Validation F1 Macro was 0.9356. "
        "- **What to improve:** Tune hyperparameters and try boosting. "
        "**Format constraints:** - Exactly four short markdown paragraphs."
    )

    out = run_summary.normalize_summary_text(raw)

    assert out.startswith("**What was built** — A sklearn Pipeline with StandardScaler and RandomForestClassifier.")
    assert "**How it was solved** — Inspected data, trained on train_df, validated on val_df." in out
    assert "Analyze the submitted solution" not in out
    assert "Format constraints" not in out


def test_normalize_summary_preserves_complete_sections_when_total_is_short():
    raw = (
        "**What was built** — A sklearn Pipeline containing StandardScaler and RandomForestClassifier.\n"
        "**How it was solved** — Inspected data structure and distribution. Separated features and target. "
        "Created a pipeline with scaling and a random forest. Trained on `train_df`. Evaluated on `val_df`.\n"
        "**Result** — Validation F1 Macro was 0.9356.\n"
        "**What to improve** — Since it was a baseline, improvements could include hyperparameter tuning, "
        "trying other models, or additional feature engineering."
    )

    out = run_summary.normalize_summary_text(raw)

    assert out.endswith("additional feature engineering.")
    assert "…" not in out


def test_fallback_summary_from_solution_uses_validation_metric():
    out = run_summary.fallback_summary_from_solution(
        (
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "X_train = train_df.drop(columns=[target_col])\n"
            "numeric_features = X_train.columns.tolist()\n"
            "model = Pipeline(steps=[('prep', ColumnTransformer([('num', StandardScaler(), numeric_features)])),"
            " ('clf', RandomForestClassifier())])\n"
        ),
        validation_metric=0.9355934,
    )

    assert "**Result** — Best validation metric was 0.9356." in out
    assert "StandardScaler" in out


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


def test_read_solution_code_reads_exported_notebook_python(tmp_path):
    assert run_summary.read_solution_code(tmp_path) is None

    exported = tmp_path / run_summary.FINAL_SOLUTION_FILENAME
    exported.write_text("model = fitted_pipeline\n", encoding="utf-8")

    assert run_summary.read_solution_code(tmp_path) == "model = fitted_pipeline"


def test_generate_and_write_failed_call_writes_nothing(tmp_path):
    out = run_summary.generate_and_write(FakeClient(raises=True), "m", tmp_path, conversation=[])
    assert out is None
    assert not (tmp_path / run_summary.SUMMARY_FILENAME).exists()

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from dashboard.server.app.config import REPO_ROOT, default_python_bin
from dashboard.server.app.services import mlflow_store, run_launcher


def _fake_run(params: dict, metrics: dict, status: str = "FINISHED"):
    return SimpleNamespace(
        data=SimpleNamespace(
            params=params,
            metrics=metrics,
            tags={"mlflow.runName": "dash_unit"},
        ),
        info=SimpleNamespace(
            run_id="1234567890abcdef",
            status=status,
            start_time=1000,
            end_time=2000,
            experiment_id="0",
        ),
    )


def test_default_python_bin_finds_windows_venv_layout(tmp_path):
    python = tmp_path / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    assert default_python_bin(tmp_path) == str(python)


def test_default_python_bin_finds_unix_venv_layout(tmp_path):
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    assert default_python_bin(tmp_path) == str(python)


def test_default_python_bin_falls_back_to_current_interpreter(tmp_path):
    assert default_python_bin(tmp_path) == sys.executable


def test_run_record_maps_baseline_single_shot_and_infers_one_step():
    record = mlflow_store._run_record(
        _fake_run(
            params={
                "experiment_type": "baseline_single_shot",
                "model": "fake-model",
                "dataset": "unit-ds",
            },
            metrics={
                "final_test_metric": 0.75,
                "has_test_metric": 1,
                "valid_submit": 1,
            },
        )
    )

    assert record["mode"] == "single"
    assert record["status"] == "success"
    assert record["score"] == 0.75
    assert record["step"] == 1
    assert record["steps"] == 1


def test_run_record_maps_repeated_single_shot_attempt_progress():
    record = mlflow_store._run_record(
        _fake_run(
            params={
                "experiment_type": "repeated_single_shot",
                "max_attempts": "5",
                "model": "fake-model",
                "dataset": "unit-ds",
            },
            metrics={
                "final_test_metric": 0.8,
                "has_test_metric": 1,
                "valid_submit": 1,
                "attempts_used": 3,
            },
        )
    )

    assert record["mode"] == "repeated"
    assert record["status"] == "success"
    assert record["score"] == 0.8
    assert record["step"] == 3
    assert record["steps"] == 5


def test_run_record_maps_fixed_gym_to_fixed_mode():
    record = mlflow_store._run_record(
        _fake_run(
            params={
                "experiment_type": "fixed_gym",
                "model": "fake-model",
                "dataset": "unit-ds",
            },
            metrics={},
        )
    )

    assert record["mode"] == "fixed"


def test_run_record_exposes_all_batch_metadata():
    record = mlflow_store._run_record(
        _fake_run(
            params={
                "experiment_type": "directive_gym",
                "requested_mode": "all",
                "batch_id": "batch-1",
                "product_mode": "directive_gym",
                "mode_label": "directive_gym",
                "mode_order": "3",
                "model": "fake-model",
                "dataset": "unit-ds",
            },
            metrics={},
        )
    )

    assert record["requestedMode"] == "all"
    assert record["batchId"] == "batch-1"
    assert record["productMode"] == "directive_gym"
    assert record["modeOrder"] == 3


def test_run_record_hides_placeholder_zero_score_for_failed_submit():
    record = mlflow_store._run_record(
        _fake_run(
            params={"experiment_type": "directive_gym"},
            metrics={
                "test_metric": 0.0,
                "has_test_metric": 0,
                "submit_failed": 1,
            },
        )
    )

    assert record["status"] == "failed"
    assert record["score"] is None


def test_checklist_uses_authoritative_fallback_when_episode_artifacts_missing():
    data = mlflow_store.checklist(None, fallback_coverage=0.88)

    assert data["coverage"] == 0.88
    assert data["closed"] == round(0.88 * data["total"])


def test_checklist_prefers_private_artifact_item_identity(tmp_path):
    episode = tmp_path / "artifacts" / "episode"
    episode.mkdir(parents=True)
    private = tmp_path / "artifacts" / "private_episode"
    private.mkdir()
    (episode / "notebook_events.json").write_text(
        json.dumps(
            [
                {
                    "step": 1,
                    "cell_id": "cell_01",
                    "source_after": "train_df.isna().sum()",
                    "execution_result": {
                        "success": True,
                        "stdout": "missing values: 0",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    (private / "checklist_private.json").write_text(
        json.dumps(
            {
                "covered": ["task_understanding", "target_exclusion", "validation_evaluated"],
                "coverage": 0.25,
                "evidence": [
                    {"key": "task_understanding", "step": 1, "reason": "read task prompt", "cell_id": "cell_01"},
                    {"key": "target_exclusion", "step": 2},
                    {"key": "validation_evaluated", "step": 3},
                ],
            }
        ),
        encoding="utf-8",
    )

    data = mlflow_store.checklist(episode, target_col="target", fallback_coverage=0.75)
    closed = {item["id"] for item in data["items"] if item["closed"]}

    assert data["coverage"] == 0.25
    assert data["closed"] == 3
    assert data["knownClosed"] == 3
    assert closed == {"task_understanding", "target_exclusion", "validation_evaluated"}
    assert "missing_values_audit" not in closed
    task_item = next(item for item in data["items"] if item["id"] == "task_understanding")
    assert task_item["evidence"] == [{"step": 1, "reason": "read task prompt", "cellId": "cell_01"}]


def test_checklist_reads_run_gym_episode_private_artifact_dir(tmp_path):
    episode = tmp_path / "live" / "workspace"
    episode.mkdir(parents=True)
    artifact_dir = tmp_path / "mlflow" / "artifacts"
    private = artifact_dir / "episode_private"
    private.mkdir(parents=True)
    (private / "checklist_private.json").write_text(
        json.dumps(
            {
                "covered": [
                    "task_understanding",
                    "schema_review",
                    "target_exclusion",
                    "reproducible_solution",
                    "baseline_candidate_created",
                    "validation_evaluated",
                    "submit_ready_artifact",
                ],
                "coverage": 0.58,
                "evidence": [
                    {"key": "task_understanding", "step": 1},
                    {"key": "schema_review", "step": 1},
                    {"key": "target_exclusion", "step": 1},
                    {"key": "reproducible_solution", "step": 3},
                    {"key": "baseline_candidate_created", "step": 4},
                    {"key": "validation_evaluated", "step": 4},
                    {"key": "submit_ready_artifact", "step": 4},
                ],
            }
        ),
        encoding="utf-8",
    )

    data = mlflow_store.checklist(
        episode,
        target_col="target",
        fallback_coverage=0.25,
        artifact_dir=artifact_dir,
    )

    assert data["coverage"] == 0.58
    assert data["closed"] == 7
    assert data["knownClosed"] == 7
    submit_item = next(item for item in data["items"] if item["id"] == "submit_ready_artifact")
    assert submit_item["evidence"] == [{"step": 4, "reason": "", "cellId": None}]


def test_checklist_replay_beats_stale_zero_fallback(tmp_path):
    (tmp_path / "notebook_events.json").write_text(
        json.dumps(
            [
                {
                    "step": 1,
                    "cell_id": "cell_01",
                    "source_after": """
target_col = "target"
print(train_df.columns)
cat_cols = train_df.select_dtypes(include=["object", "category"]).columns
X_train = train_df.drop(columns=[target_col])
from sklearn.preprocessing import OneHotEncoder
""",
                    "execution_result": {"success": True, "stdout": ""},
                }
            ]
        ),
        encoding="utf-8",
    )

    data = mlflow_store.checklist(tmp_path, target_col="target", fallback_coverage=0.0)
    closed = {item["id"] for item in data["items"] if item["closed"]}

    assert data["coverage"] > 0.0
    assert data["closed"] == round(data["coverage"] * data["total"])
    assert closed == {
        "task_understanding",
        "schema_review",
        "categorical_features_audit",
        "target_exclusion",
    }


def test_checklist_reads_legacy_generated_solution_artifact(tmp_path):
    (tmp_path / "generated_solution.py").write_text(
        """
target_col = "target"
numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
X_train = train_df.drop(columns=[target_col])
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
model = GridSearchCV(Pipeline([]), param_grid={"classifier__n_estimators": [100]})
""",
        encoding="utf-8",
    )
    (tmp_path / "stdout.txt").write_text("", encoding="utf-8")

    data = mlflow_store.checklist(
        None,
        target_col="target",
        fallback_coverage=0.0,
        artifact_dir=tmp_path,
    )
    closed = {item["id"] for item in data["items"] if item["closed"]}

    assert data["coverage"] > 0.0
    assert closed == {
        "task_understanding",
        "schema_review",
        "categorical_features_audit",
        "target_exclusion",
    }


def test_episode_artifacts_expose_current_stage_type_and_thoughts(tmp_path):
    (tmp_path / "episode_summary.json").write_text(
        json.dumps({"current_stage": "validation_analysis"}),
        encoding="utf-8",
    )
    (tmp_path / "feedback_trace.json").write_text(
        json.dumps(
            [
                {
                    "type": "think",
                    "step": 0,
                    "budget_remaining": 20,
                    "stage": "planning",
                    "thoughts": "I will inspect data, validate a candidate, and submit only when ready.",
                    "stdout": "[THINK] Thought recorded.",
                    "feedback_items": [],
                },
                {
                    "type": "add_cell",
                    "step": 1,
                    "budget_remaining": 19,
                    "stage": "feature_pipeline_building",
                    "thoughts": "I am building a reproducible preprocessing pipeline.",
                    "code": "model = pipeline",
                    "feedback_items": [{"channel": "runtime", "message": "ok"}],
                },
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "scratchpad.json").write_text(
        json.dumps(
            [
                {
                    "step": 1,
                    "type": "think",
                    "stage": "planning",
                    "thoughts": "I will inspect data first.",
                    "timestamp": "2026-06-05T00:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "notebook_events.json").write_text(
        json.dumps(
            [
                {"type": "think", "step": 0, "stage": "planning", "non_mutating": True},
                {"type": "add_cell", "step": 1, "stage": "feature_pipeline_building"},
            ]
        ),
        encoding="utf-8",
    )

    assert mlflow_store.current_stage(tmp_path) == "validation_analysis"

    trajectory = mlflow_store.trajectory(tmp_path)
    assert trajectory[0]["type"] == "think"
    assert trajectory[0]["stage"] == "planning"
    assert trajectory[0]["thoughts"].startswith("I will inspect")
    assert "action" not in trajectory[0]
    assert trajectory[1]["type"] == "add_cell"
    assert trajectory[1]["stage"] == "feature_pipeline_building"
    assert trajectory[1]["thoughts"] == "I am building a reproducible preprocessing pipeline."

    thoughts = mlflow_store.thoughts(tmp_path)
    assert thoughts == [
        {
            "step": 1,
            "type": "think",
            "stage": "planning",
            "thoughts": "I will inspect data first.",
            "timestamp": "2026-06-05T00:00:00Z",
        }
    ]

    progress = mlflow_store.episode_progress(tmp_path)
    assert progress["currentStage"] == "validation_analysis"


def test_run_summary_reader_maps_fields_and_handles_missing(tmp_path):
    # Missing file → empty dict (old runs hide the summary card).
    assert mlflow_store.run_summary(tmp_path) == {}
    assert mlflow_store.has_run_summary(tmp_path) is False

    (tmp_path / "run_summary.json").write_text(
        json.dumps({
            "summary": "  **What was built** — gradient boosting.  ",
            "model": "llama-3.1-8b-instant",
            "generated_at": "2026-06-06T00:00:00Z",
            "input_tokens": 120,
        }),
        encoding="utf-8",
    )
    out = mlflow_store.run_summary(tmp_path)
    assert out == {
        "summary": "**What was built** — gradient boosting.",
        "model": "llama-3.1-8b-instant",
        "generatedAt": "2026-06-06T00:00:00Z",
    }
    assert mlflow_store.has_run_summary(tmp_path) is True


def test_run_summary_reader_ignores_blank_summary(tmp_path):
    (tmp_path / "run_summary.json").write_text(
        json.dumps({"summary": "   ", "model": "m"}), encoding="utf-8"
    )
    assert mlflow_store.run_summary(tmp_path) == {}


def test_run_summary_reader_normalizes_verbose_model_output(tmp_path):
    (tmp_path / "run_summary.json").write_text(
        json.dumps(
            {
                "summary": (
                    "The user wants a retrospective summary.\n\n"
                    "Constraint checklist & Confidence score:\n"
                    "1. Past tense? Yes.\n\n"
                    "What was built: Built a sklearn Pipeline with LightGBM.\n"
                    "How it was solved: Used numeric features and checked the model on val_df.\n"
                    "Result: Achieved F1 Macro = 0.9411.\n"
                    "What to improve: Tune hyperparameters.\n"
                ),
                "model": "m",
            }
        ),
        encoding="utf-8",
    )

    out = mlflow_store.run_summary(tmp_path)
    assert out["summary"].startswith("**What was built** — Built a sklearn Pipeline with LightGBM.")
    assert "Constraint checklist" not in out["summary"]


def test_run_summary_reader_falls_back_to_solution_code_for_unusable_saved_summary(tmp_path):
    (tmp_path / "run_summary.json").write_text(
        json.dumps(
            {
                "summary": (
                    "* **Context:** already submitted solution.\n"
                    "* **Constraints:** no bullets.\n"
                    "3. **Drafting the Sections:**\n"
                    "**What was built** I built a model.\n"
                ),
                "model": "m",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "final_notebook.py").write_text(
        "\n".join(
            [
                "from sklearn.pipeline import Pipeline",
                "from sklearn.preprocessing import StandardScaler",
                "from sklearn.ensemble import RandomForestClassifier",
                "X_train = train_df.drop(columns=[target_col])",
                "model = Pipeline(steps=[('scale', StandardScaler()), ('clf', RandomForestClassifier())])",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "episode_summary.json").write_text(
        json.dumps({"best_validation_metric": 0.9355934}),
        encoding="utf-8",
    )

    out = mlflow_store.run_summary(tmp_path)

    assert out["summary"].startswith("**What was built** — Built a scikit-learn Pipeline around RandomForestClassifier")
    assert "**Result** — Best validation metric was 0.9356." in out["summary"]
    assert "Constraints" not in out["summary"]


def test_run_detail_frontend_shows_summary_card_and_gates_tab():
    source = (REPO_ROOT / "dashboard" / "web" / "src" / "pages" / "RunDetail.tsx").read_text(
        encoding="utf-8"
    )
    # Thoughts tab is shown for runs with a scratchpad OR a self-summary.
    assert "run.thoughtsEnabled || !!run.hasSummary" in source
    assert "api.runSummary(id)" in source
    assert "SummaryCard" in source
    assert "parseSummarySections" in source
    assert "run-summary-section-body" in source


def test_run_detail_frontend_declares_stage_and_think_rendering():
    source = (REPO_ROOT / "dashboard" / "web" / "src" / "pages" / "RunDetail.tsx").read_text(
        encoding="utf-8"
    )

    assert 'label="Этап"' in source
    assert 'think: "мысль"' in source
    assert 'planning: "Планирование"' in source
    assert "stageLabel(s.stage)" in source
    # Thoughts are shown only on the «Мысли» tab, never inline in the trajectory.
    assert "s.thoughts" not in source


def test_new_run_frontend_limits_thoughts_toggle_to_gym_modes():
    source = (REPO_ROOT / "dashboard" / "web" / "src" / "pages" / "NewRun.tsx").read_text(
        encoding="utf-8"
    )

    assert 'const thoughtsSupported = selectedModes.some(m => m === "directive" || m === "free" || m === "fixed");' in source


def test_run_launcher_planned_steps_match_dashboard_modes():
    assert run_launcher._planned_steps({"mode": "single"}) == 1
    assert run_launcher._planned_steps({"mode": "repeated", "shots": 4}) == 4
    assert run_launcher._planned_steps({"mode": "directive", "maxSteps": 8}) == 8
    assert run_launcher._planned_steps({"mode": "free", "maxSteps": 6}) == 6
    assert run_launcher._planned_steps({"mode": "fixed", "maxSteps": 12}) == 12
    assert run_launcher._planned_steps({"mode": "batch", "modes": ["single", "repeated", "directive"]}) == 3
    assert run_launcher._planned_steps({"mode": "all"}) == 5


def test_run_launcher_batch_mode_uses_common_runner_with_selected_modes(monkeypatch):
    monkeypatch.setattr(
        run_launcher.model_store,
        "get_model",
        lambda model_id: {"id": model_id, "name": "fake-model", "maxTokens": 4096},
    )
    args = run_launcher._runner_args(
        {
            "mode": "batch",
            "modes": ["single", "repeated", "free", "directive", "fixed"],
            "budgetMode": "local",
            "modelId": "fake",
            "runName": "dash_live_unit",
            "maxSteps": 8,
            "shots": 4,
        }
    )

    assert args[:9] == [
        "-m",
        "experiments.run",
        "--modes",
        "single_shot",
        "repeated_single_shot",
        "free_gym",
        "directive_gym",
        "fixed_gym",
        "--budget-mode",
    ]
    assert "local" in args
    assert "--max-steps" in args
    assert "--shots" in args


def test_run_launcher_fixed_mode_uses_fixed_runner(monkeypatch):
    monkeypatch.setattr(
        run_launcher.model_store,
        "get_model",
        lambda model_id: {"id": model_id, "name": "fake-model", "maxTokens": 4096},
    )
    args = run_launcher._runner_args(
        {
            "mode": "fixed",
            "budgetMode": "local",
            "modelId": "fake",
            "runName": "dash_live_unit",
            "maxSteps": 8,
            "enableThoughts": True,
        }
    )

    assert args[:2] == ["-m", "experiments.run_fixed"]
    assert "--max-steps" in args
    assert "--enable-thoughts" in args


def test_run_launcher_supports_thoughts_for_gym_modes():
    assert run_launcher._supports_thoughts("directive") is True
    assert run_launcher._supports_thoughts("free") is True
    assert run_launcher._supports_thoughts("fixed") is True
    assert run_launcher._supports_thoughts("gym") is True
    assert run_launcher._supports_thoughts("iterative") is True
    assert run_launcher._supports_thoughts("single") is False


def test_run_launcher_uses_model_name_from_model_id(monkeypatch):
    monkeypatch.setattr(
        run_launcher.model_store,
        "get_model",
        lambda model_id: {
            "id": model_id,
            "name": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "provider": "OpenAI-совместимый",
            "baseUrl": "https://openrouter.ai/api/v1",
            "maxTokens": 4096,
        },
    )

    args = run_launcher._runner_args(
        {
            "mode": "directive",
            "budgetMode": "local",
            "modelId": "openrouter",
            "runName": "dash_live_unit",
            "maxSteps": 8,
        }
    )

    assert "--model" in args
    assert args[args.index("--model") + 1] == "nvidia/nemotron-3-ultra-550b-a55b:free"


def test_run_launcher_requires_model_id():
    with pytest.raises(ValueError, match="Model must be selected"):
        run_launcher._runner_args(
            {
                "mode": "directive",
                "budgetMode": "local",
                "runName": "dash_live_unit",
                "maxSteps": 8,
            }
        )


def test_run_launcher_llm_env_sets_provider_from_selected_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "google")

    monkeypatch.setattr(
        run_launcher.model_store,
        "get_model",
        lambda model_id: {
            "id": model_id,
            "name": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "provider": "OpenAI-compatible",
            "baseUrl": "https://openrouter.ai/api/v1",
            "apiKey": "sk-test",
        },
    )

    env = run_launcher._llm_env({"modelId": "openrouter"})

    assert env["LLM_PROVIDER"] == "openai"
    assert env["LLM_MODEL"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert env["LLM_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert env["LLM_API_KEY"] == "sk-test"


def test_run_launcher_llm_env_maps_gemini_and_litellm_providers(monkeypatch):
    records = {
        "gemini": {"provider": "Gemini", "name": "gemini-2.5-flash"},
        "lite": {"provider": "LiteLLM", "name": "groq/llama-3.3-70b-versatile"},
    }
    monkeypatch.setattr(
        run_launcher.model_store,
        "get_model",
        lambda model_id: records[model_id],
    )

    assert run_launcher._llm_env({"modelId": "gemini"})["LLM_PROVIDER"] == "google"
    assert run_launcher._llm_env({"modelId": "lite"})["LLM_PROVIDER"] == "litellm"


def test_run_launcher_gemini_model_does_not_need_base_url(monkeypatch):
    monkeypatch.setattr(
        run_launcher.model_store,
        "get_model",
        lambda model_id: {
            "id": model_id,
            "name": "gemini-2.5-flash",
            "provider": "Gemini",
            "apiKey": "gemini-key",
            "baseUrl": "",
        },
    )

    env = run_launcher._llm_env({"modelId": "gemini"})

    assert env["LLM_PROVIDER"] == "google"
    assert env["LLM_MODEL"] == "gemini-2.5-flash"
    assert env["GEMINI_API_KEY"] == "gemini-key"
    assert "LLM_BASE_URL" not in env
    assert "LLM_API_KEY" not in env


def test_run_launcher_python_available_accepts_paths_and_rejects_missing(tmp_path):
    assert run_launcher._python_available(sys.executable)
    assert not run_launcher._python_available(str(tmp_path / "missing-python"))

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from dashboard.server.app.config import REPO_ROOT, default_python_bin
from dashboard.server.app.services import mlflow_store, run_launcher


def _fake_run(params: dict, metrics: dict, status: str = "FINISHED", tags: dict | None = None):
    merged_tags = {"mlflow.runName": "dash_unit"}
    if tags:
        merged_tags.update(tags)
    return SimpleNamespace(
        data=SimpleNamespace(
            params=params,
            metrics=metrics,
            tags=merged_tags,
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


def test_run_record_maps_fixed_transitions_to_fixed_mode():
    record = mlflow_store._run_record(
        _fake_run(
            params={
                "experiment_type": "fixed_transitions",
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
                "experiment_type": "gym_with_checklist",
                "requested_mode": "all",
                "batch_id": "batch-1",
                "product_mode": "gym_with_checklist",
                "mode_label": "gym_with_checklist",
                "mode_order": "3",
                "model": "fake-model",
                "dataset": "unit-ds",
            },
            metrics={},
        )
    )

    assert record["requestedMode"] == "all"
    assert record["batchId"] == "batch-1"
    assert record["productMode"] == "gym_with_checklist"
    assert record["modeOrder"] == 3


def test_run_record_hides_placeholder_zero_score_for_failed_submit():
    record = mlflow_store._run_record(
        _fake_run(
            params={"experiment_type": "gym_with_checklist"},
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


def test_new_run_frontend_limits_thoughts_toggle_to_gym_and_iterative():
    source = (REPO_ROOT / "dashboard" / "web" / "src" / "pages" / "NewRun.tsx").read_text(
        encoding="utf-8"
    )

    assert 'const thoughtsSupported = selectedModes.some((m) => m === "gym" || m === "iterative");' in source


def test_run_launcher_planned_steps_match_dashboard_modes():
    assert run_launcher._planned_steps({"mode": "single"}) == 1
    assert run_launcher._planned_steps({"mode": "repeated", "shots": 4}) == 4
    assert run_launcher._planned_steps({"mode": "gym", "maxSteps": 8}) == 8
    assert run_launcher._planned_steps({"mode": "iterative", "maxSteps": 6}) == 6
    assert run_launcher._planned_steps({"mode": "fixed", "maxSteps": 12}) == 12
    assert run_launcher._planned_steps({"mode": "batch", "modes": ["single", "repeated", "gym"]}) == 3
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
            "modes": ["single", "repeated", "iterative", "gym", "fixed"],
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
        "iterative_no_checklist",
        "gym_with_checklist",
        "fixed_transitions",
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
    assert "--enable-thoughts" not in args


def test_run_launcher_supports_thoughts_only_for_gym_and_iterative():
    assert run_launcher._supports_thoughts("gym") is True
    assert run_launcher._supports_thoughts("iterative") is True
    assert run_launcher._supports_thoughts("fixed") is False
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
            "mode": "gym",
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
                "mode": "gym",
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


def test_run_record_exposes_prompt_preset_tags():
    """run_gym/run_fixed log prompt_preset_id and prompt_sha256 as MLflow tags;
    the dashboard surfaces them in the run record so the UI can show which
    preset was used (and whether it was the controlled-default baseline)."""
    record = mlflow_store._run_record(
        _fake_run(
            params={"experiment_type": "gym_with_checklist", "model": "m", "dataset": "d"},
            metrics={},
            tags={
                "prompt_preset_id": "minimal",
                "prompt_sha256": "abc123",
                "prompt_default": "false",
            },
        )
    )
    assert record["promptPresetId"] == "minimal"
    assert record["promptSha256"] == "abc123"
    assert record["promptIsDefault"] is False


def test_run_record_marks_default_preset_when_logged():
    record = mlflow_store._run_record(
        _fake_run(
            params={"experiment_type": "gym_with_checklist", "model": "m", "dataset": "d"},
            metrics={},
            tags={"prompt_preset_id": "default", "prompt_default": "true"},
        )
    )
    assert record["promptPresetId"] == "default"
    assert record["promptIsDefault"] is True


def test_run_record_legacy_run_has_null_preset_fields():
    """Runs that predate this PR have no preset tag — show null, NOT 'default'.
    Mislabeling them as default would falsely imply they used today's prompt."""
    record = mlflow_store._run_record(
        _fake_run(
            params={"experiment_type": "gym_with_checklist", "model": "m", "dataset": "d"},
            metrics={},
        )
    )
    assert record["promptPresetId"] is None
    assert record["promptSha256"] is None
    assert record["promptIsDefault"] is None

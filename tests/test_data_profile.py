import json
import subprocess

import pandas as pd

from gym.data_profile import (
    build_compact_profile,
    extract_ydata_summary,
    format_profile_for_agent,
    run_ydata_profile,
    run_ydata_profile_subprocess,
)


def test_compact_profile_respects_max_chars_and_uses_train_val_only():
    train = pd.DataFrame({"x": [1, 2, None], "cat": ["a", "b", "a"], "target": [0, 1, 0]})
    val = pd.DataFrame({"x": [3], "cat": ["c"], "target": [1]})

    profile = build_compact_profile(train, val, "target", "accuracy")
    text = format_profile_for_agent(profile, max_chars=800)

    assert "[DATA PROFILE]" in text
    assert profile["unseen_categories_in_val"][0]["column"] == "cat"
    assert len(text) <= 820


def test_ydata_summary_extraction_handles_mock_json():
    summary = extract_ydata_summary(
        {
            "table": {"n": 10, "n_var": 3, "n_cells_missing": 2, "n_duplicates": 1},
            "variables": {
                "id": {"type": "Numeric", "n": 10, "n_distinct": 10, "p_missing": 0.0},
                "mostly_missing": {"type": "Categorical", "n": 10, "n_distinct": 2, "p_missing": 0.5},
                "constant": {"type": "Numeric", "n": 10, "n_distinct": 1, "p_missing": 0.0},
            },
            "alerts": ["constant column"],
        }
    )

    assert summary["n_rows"] == 10
    assert "mostly_missing" in summary["high_missing_columns"]
    assert "constant" in summary["constant_columns"]
    assert "id" in summary["high_cardinality_columns"]


def test_ydata_unavailable_returns_graceful_fallback(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(
                {
                    "available": False,
                    "success": False,
                    "error_type": "ImportError",
                    "error_message": "ydata-profiling is not installed",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("gym.data_profile.subprocess.run", fake_run)
    result = run_ydata_profile(
        pd.DataFrame({"x": [1, 2], "target": [0, 1]}),
        "target",
        tmp_path,
        max_rows=10,
        max_cols=10,
        timeout_sec=1,
    )

    assert result["success"] is False
    assert result["error_type"] == "ImportError"


def test_ydata_timeout_returns_fallback_without_lingering_input(tmp_path, monkeypatch):
    def fake_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))

    monkeypatch.setattr("gym.data_profile.subprocess.run", fake_timeout)

    result = run_ydata_profile_subprocess(
        pd.DataFrame({"x": [1, 2], "target": [0, 1]}),
        "target",
        tmp_path,
        max_rows=10,
        max_cols=10,
        timeout_sec=1,
    )

    assert result["success"] is False
    assert result["timed_out"] is True
    assert result["error_type"] == "TimeoutExpired"
    assert result["rows_profiled"] == 2
    assert result["cols_profiled"] == 2
    assert not list(tmp_path.glob("autovibe_ydata_input_*.csv"))

"""Tests for notebook-style CellHistory."""
from gym import CellHistory, Observation


def test_append_observation_creates_cell():
    history = CellHistory()
    observation = Observation(
        action="code",
        step=1,
        budget_remaining=2,
        code="x = 1\nprint(x)",
        stdout="1\n",
        checklist_coverage=0.25,
    )

    cell = history.append_observation(observation)

    assert len(history) == 1
    assert cell.execution_count == 1
    assert cell.action == "code"
    assert cell.code == "x = 1\nprint(x)"
    assert cell.stdout == "1\n"


def test_reset_clears_cells():
    history = CellHistory()
    history.append_observation(
        Observation(action="code", step=1, budget_remaining=0, code="pass")
    )

    history.reset()

    assert len(history) == 0
    assert history.last() is None


def test_feedback_context_contains_recent_cells():
    history = CellHistory()
    history.append_observation(
        Observation(
            action="code",
            step=1,
            budget_remaining=1,
            code="value = 41",
        )
    )
    history.append_observation(
        Observation(
            action="code",
            step=2,
            budget_remaining=0,
            code="print(value + 1)",
            stdout="42\n",
        )
    )

    context = history.to_feedback_context(max_cells=2)

    assert "[NOTEBOOK]" in context
    assert "In [1] code" in context
    assert "In [2] code" in context
    assert "print(value + 1)" in context
    assert "42" in context


def test_to_markdown_includes_submit_result():
    history = CellHistory()
    history.append_observation(
        Observation(
            action="submit",
            step=3,
            budget_remaining=0,
            submitted=True,
            done=True,
            test_metric=0.9,
            model_var="best_model",
        )
    )

    markdown = history.to_markdown()

    assert "# AutoVibe Gym Cell History" in markdown
    assert "Submitted model: best_model" in markdown
    assert "Test metric: 0.9" not in markdown

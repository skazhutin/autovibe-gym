import pytest

from gym.protocol import Action, ActionParseError, Observation, coerce_action


def test_action_accepts_aliases_and_defaults_submit_model_name():
    assert Action.from_payload({"action": "run-code", "code": "print(1)"}).type == "code"
    submit = Action.from_payload({"type": "submit_model"})
    assert submit.type == "submit"
    assert submit.model_var == "model"


def test_action_rejects_missing_type_and_missing_code():
    with pytest.raises(ActionParseError, match="missing the 'type'"):
        Action.from_payload({})
    with pytest.raises(ActionParseError, match="missing the 'code'"):
        Action.from_payload({"type": "code"})


def test_action_parses_json_and_markdown_json_blocks():
    action = Action.from_llm_response(
        '```json\n{"type": "code", "code": "x = 1"}\n```'
    )

    assert action == Action.code_action("x = 1")


def test_action_falls_back_to_python_code_block_for_legacy_responses():
    action = Action.from_llm_response("```python\nx = 1\n```")

    assert action == Action.code_action("x = 1")


def test_action_parses_legacy_submit_literal():
    assert Action.from_llm_response("SUBMIT") == Action.submit_action("model")


def test_coerce_action_rejects_unknown_object_type():
    with pytest.raises(ActionParseError, match="Unsupported action object"):
        coerce_action(123)


def test_observation_feedback_includes_outputs_hints_and_submission():
    observation = Observation(
        action="submit",
        step=2,
        budget_remaining=0,
        stdout="done",
        stderr="warn",
        hints=["hint"],
        checklist_coverage=0.5,
        submitted=True,
        test_metric=0.9,
    )

    feedback = observation.to_feedback_message()

    assert "[EXECUTION RESULT]\ndone" in feedback
    assert "[CONTRACT FEEDBACK]\nwarn" in feedback
    assert "- hint" in feedback
    assert "test_metric=0.9" not in feedback
    assert "[SUBMITTED] Final candidate accepted" in feedback


def test_observation_feedback_marks_budget_exhaustion_without_submission():
    feedback = Observation(
        action="code",
        step=3,
        budget_remaining=0,
        done=True,
    ).to_feedback_message()

    assert "[DONE] Step budget exhausted." in feedback


def test_notebook_actions_parse_from_json_payloads():
    add = Action.from_payload(
        {
            "type": "add_cell",
            "cell_type": "code",
            "source": "print(train_df.shape)",
            "execute": True,
        }
    )
    assert add.type == "add_cell"
    assert add.source == "print(train_df.shape)"
    assert add.execute is True

    update = Action.from_payload(
        {
            "type": "update_cell",
            "cell_id": "cell_01",
            "source": "print(val_df.shape)",
        }
    )
    assert update.type == "update_cell"
    assert update.cell_id == "cell_01"

    move = Action.from_payload(
        {"type": "move_cell", "cell_id": "cell_01", "new_position": 0}
    )
    assert move.new_position == 0

    assert Action.from_payload({"type": "restart_and_run_all"}).type == "restart_and_run_all"
    assert Action.from_payload({"type": "validate"}).model_var == "model"

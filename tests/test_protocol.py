import pytest

from gym.protocol import (
    AGENT_ACTION_TYPE_VALUES,
    AGENT_STAGE_VALUES,
    ACTION_JSON_SCHEMA,
    Action,
    ActionParseError,
    Observation,
    coerce_action,
)


def test_action_rejects_alternate_type_fields_and_unknown_type_values():
    with pytest.raises(ActionParseError, match="canonical 'type'"):
        Action.from_payload({"action": "run-code", "code": "print(1)"})
    with pytest.raises(ActionParseError, match="Unsupported action type"):
        Action.from_payload({"type": "submit_model"})


def test_action_defaults_submit_model_name():
    submit = Action.from_payload({"type": "submit", "stage": "submission"})
    assert submit.type == "submit"
    assert submit.model_var == "model"


def test_action_rejects_missing_type_and_missing_code():
    with pytest.raises(ActionParseError, match="missing the 'type'"):
        Action.from_payload({})
    with pytest.raises(ActionParseError, match="missing the 'code'"):
        Action.from_payload({"type": "code"})


def test_action_parses_json_and_markdown_json_blocks():
    action = Action.from_llm_response(
        '```json\n{"type": "code", "stage": "feature_pipeline_building", "code": "x = 1"}\n```'
    )

    assert action == Action.code_action("x = 1")


def test_action_rejects_malformed_or_non_object_json():
    with pytest.raises(ActionParseError, match="Invalid action JSON"):
        Action.from_json('{"type": "code",')
    with pytest.raises(ActionParseError, match="must be an object"):
        Action.from_json('["not", "an", "object"]')


def test_action_falls_back_to_python_code_block_for_legacy_responses():
    action = Action.from_llm_response("```python\nx = 1\n```")

    assert action == Action.code_action("x = 1")


def test_action_parses_legacy_submit_literal():
    assert Action.from_llm_response("SUBMIT") == Action.submit_action("model")


def test_action_strips_trailing_tool_call_token():
    action = Action.from_llm_response(
        '{"type": "inspect_notebook", "stage": "reproducibility_check"}<tool_call|>'
    )
    assert action.type == "inspect_notebook"


def test_action_parses_json_with_tool_call_prefix():
    action = Action.from_llm_response(
        '<|tool_call>call:{"type": "submit", "stage": "submission", "model_var": "model"}'
    )
    assert action == Action.submit_action("model")


def test_action_extracts_json_object_with_surrounding_prose():
    action = Action.from_llm_response(
        'Here is my action:\n{"type": "code", "stage": "feature_pipeline_building", "code": "x = 1"}\nThanks!'
    )
    assert action == Action.code_action("x = 1")


def test_action_raises_instead_of_dumping_malformed_json_as_code():
    with pytest.raises(ActionParseError):
        Action.from_llm_response('{"type": "code", "code": "x = 1"')


def test_coerce_action_rejects_unknown_object_type():
    with pytest.raises(ActionParseError, match="Unsupported action object"):
        coerce_action(123)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"type": "dance"}, "Unsupported action type"),
        ({"action_type": "code", "code": "x = 1"}, "canonical 'type'"),
        ({"type": "code", "code": "x = 1", "notes": "old"}, "canonical 'thoughts'"),
        ({"type": "add_cell", "cell_type": "raw", "source": ""}, "cell_type"),
        ({"type": "update_cell", "source": "x = 1"}, "cell_id"),
        ({"type": "delete_cell"}, "cell_id"),
        ({"type": "run_cell"}, "cell_id"),
        ({"type": "move_cell", "cell_id": "cell_01"}, "new_position"),
        ({"type": "move_cell", "cell_id": "cell_01", "new_position": "later"}, "integer"),
        ({"type": "validate", "model_var": ""}, "model_var"),
        ({"type": "submit", "model_var": ""}, "model_var"),
    ],
)
def test_invalid_notebook_actions_fail_clearly(payload, message):
    with pytest.raises(ActionParseError, match=message):
        Action.from_payload(payload)


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


def test_observation_public_dict_omits_private_metrics():
    data = Observation(
        action="submit",
        step=1,
        budget_remaining=0,
        checklist_coverage=0.75,
        test_metric=0.91,
        submitted=True,
    ).to_dict()

    assert "test_metric" not in data
    assert "checklist_coverage" not in data
    assert data["type"] == "submit"
    assert "action" not in data


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
            "stage": "feature_pipeline_building",
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
            "stage": "feature_pipeline_building",
            "cell_id": "cell_01",
            "source": "print(val_df.shape)",
        }
    )
    assert update.type == "update_cell"
    assert update.cell_id == "cell_01"

    move = Action.from_payload(
        {
            "type": "move_cell",
            "stage": "reproducibility_check",
            "cell_id": "cell_01",
            "new_position": 0,
        }
    )
    assert move.new_position == 0

    assert Action.from_payload(
        {"type": "restart_and_run_all", "stage": "reproducibility_check"}
    ).type == "restart_and_run_all"
    assert Action.from_payload({"type": "validate", "stage": "validation_analysis"}).model_var == "model"


def test_new_gym_tool_actions_parse_and_round_trip():
    payloads = [
        {"type": "inspect_data", "stage": "data_schema_inspection"},
        {"type": "profile_data", "stage": "data_quality_inspection", "profile": "ydata"},
        {"type": "list_candidates", "stage": "candidate_training"},
        {"type": "check_candidate", "stage": "validation_analysis", "model_var": "auto"},
        {"type": "quick_validate", "stage": "validation_analysis", "model_var": "auto"},
        {
            "type": "cleanlab_diagnose",
            "stage": "validation_analysis",
            "model_var": "auto",
            "source": "validation_or_cv",
            "max_issues": 5,
        },
        {
            "type": "tune_hyperparameters",
            "stage": "model_improvement",
            "model_var": "model",
            "search_space": {"clf__max_depth": {"type": "int", "low": 1, "high": 3}},
            "n_trials": 2,
            "timeout_sec": 5,
            "scoring": "metric",
        },
        {"type": "finalize", "stage": "submission", "model_var": "auto"},
        {"type": "think", "stage": "planning", "thoughts": "Plan the run."},
    ]

    for payload in payloads:
        action = Action.from_payload(payload)
        assert action.to_dict()["type"] == payload["type"]


def test_validate_submit_finalize_accept_auto_model_var():
    for action_type in ("validate", "submit", "finalize"):
        stage = "submission" if action_type in {"submit", "finalize"} else "validation_analysis"
        action = Action.from_payload({"type": action_type, "stage": stage, "model_var": "auto"})
        assert action.model_var == "auto"


def test_protocol_constants_and_schema_include_deterministic_fields():
    assert "think" in AGENT_ACTION_TYPE_VALUES
    assert tuple(AGENT_STAGE_VALUES) == (
        "planning",
        "data_schema_inspection",
        "target_metric_inspection",
        "data_quality_inspection",
        "leakage_split_inspection",
        "preprocessing_design",
        "feature_pipeline_building",
        "baseline_modeling",
        "candidate_training",
        "validation_analysis",
        "model_improvement",
        "reproducibility_check",
        "submission",
    )
    assert '"type": "think"' in ACTION_JSON_SCHEMA or "think" in ACTION_JSON_SCHEMA
    assert "Every JSON action must include `stage`" in ACTION_JSON_SCHEMA

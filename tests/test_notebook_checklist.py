from gym.feedback import GUIDANCE_ONLY_CHECKS, MANDATORY_CHECKS, FeedbackItem, NotebookChecklist


FORBIDDEN_HINT_PARTS = [
    "SimpleImputer",
    "OneHotEncoder",
    "GridSearchCV",
    "sklearn Pipeline",
    "There are missing",
    "Missing checklist item",
    "Coverage=",
]


def test_generic_missing_and_categorical_hints_are_not_dataset_specific():
    checklist = NotebookChecklist(target_col="target")

    hint = checklist.record_execution(
        source="print(train_df.shape)",
        stdout="(10, 4)",
        cell_id="cell_01",
        step=1,
        execution_success=True,
    )
    assert len(hint) == 1
    assert "target" in hint[0].message.lower()

    checklist.record_structural("task_understanding", reason="unit", step=1)
    checklist.record_structural("schema_review", reason="unit", step=1)
    checklist.record_structural("target_distribution_review", reason="unit", step=1)
    checklist.record_execution(
        source="train_df.isna().sum()",
        stdout="x    0",
        cell_id="cell_02",
        step=4,
        execution_success=True,
    )
    hint = checklist.record_execution(
        source="print('next')",
        stdout="next",
        cell_id="cell_03",
        step=6,
        execution_success=True,
    )

    assert len(hint) <= 1
    text = hint[0].message if hint else ""
    for forbidden in FORBIDDEN_HINT_PARTS:
        assert forbidden not in text


def test_checklist_emits_at_most_one_hint_and_respects_cooldown():
    checklist = NotebookChecklist(target_col="target")

    first = checklist.record_execution(
        source="print('hello')",
        stdout="hello",
        cell_id="cell_01",
        step=1,
        execution_success=True,
    )
    second = checklist.record_execution(
        source="print('hello')",
        stdout="hello",
        cell_id="cell_02",
        step=2,
        execution_success=True,
    )

    assert len(first) == 1
    assert second == []


def test_runtime_error_and_contract_blocker_suppress_new_checklist_hint():
    checklist = NotebookChecklist(target_col="target")

    runtime = checklist.record_execution(
        source="train_df.isna().sum()",
        stdout="",
        cell_id="cell_01",
        step=1,
        execution_success=False,
        has_runtime_error=True,
    )
    contract = checklist.record_execution(
        source="print('ok')",
        stdout="ok",
        cell_id="cell_02",
        step=3,
        execution_success=True,
        has_contract_blocker=True,
    )

    assert runtime == []
    assert contract == []


def test_failed_or_non_output_cell_does_not_satisfy_check_by_keyword_alone():
    checklist = NotebookChecklist(target_col="target")

    checklist.record_execution(
        source="train_df.isna().sum()",
        stdout="",
        cell_id="cell_01",
        step=1,
        execution_success=False,
        has_runtime_error=True,
    )
    assert "missing_values_audit" not in checklist.covered

    checklist.record_execution(
        source="train_df.isna().sum()",
        stdout="",
        cell_id="cell_02",
        step=2,
        execution_success=True,
    )
    assert "missing_values_audit" not in checklist.covered

    checklist.record_execution(
        source="train_df.isna().sum()",
        stdout="x    0",
        cell_id="cell_03",
        step=3,
        execution_success=True,
    )
    assert "missing_values_audit" in checklist.covered


def test_feedback_item_serializes_channel_metadata():
    item = FeedbackItem(
        channel="contract",
        key="clean_run_required",
        message="Run restart_and_run_all first.",
        severity="blocker",
        visible_to_agent=True,
        cell_id="cell_01",
    )

    assert item.to_dict()["channel"] == "contract"
    assert item.to_dict()["cell_id"] == "cell_01"


def test_guidance_hints_do_not_change_mandatory_coverage_denominator():
    checklist = NotebookChecklist(target_col="target")

    assert len(MANDATORY_CHECKS) == 12
    checklist.record_structural(
        GUIDANCE_ONLY_CHECKS[0],
        reason="guidance only",
        step=1,
    )
    assert checklist.coverage() == 0.0

    for key in MANDATORY_CHECKS:
        checklist.record_structural(key, reason="unit", step=2)

    assert checklist.coverage() == 1.0

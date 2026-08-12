from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


EXPECTED_ARMS = {"A", "B", "C", "D"}
EXPECTED_FAILURE_CATEGORIES = {
    "success",
    "agent_code_failure",
    "invalid_submission",
    "budget_exhausted",
    "execution_timeout",
    "sandbox_failure",
    "provider_rate_limit",
    "provider_capacity",
    "provider_auth",
    "provider_other",
    "orchestrator_failure",
}
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "protocol_id",
    "status",
    "freeze",
    "research_questions",
    "hypotheses",
    "arms",
    "unit_of_analysis",
    "matrix",
    "budget",
    "submission_policy",
    "endpoints",
    "analysis",
    "failure_policy",
    "checklist_validation",
    "exclusions",
    "todos",
}


def load_protocol(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Protocol root must be a YAML mapping.")
    return data


def validate_protocol(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    missing = sorted(REQUIRED_TOP_LEVEL - set(data))
    if missing:
        errors.append(f"missing top-level keys: {', '.join(missing)}")

    arms = data.get("arms") or {}
    if set(arms) != EXPECTED_ARMS:
        errors.append(f"arms must be exactly {sorted(EXPECTED_ARMS)}")
    else:
        confirmatory = {key for key, value in arms.items() if value.get("confirmatory") is True}
        if confirmatory != {"A", "B", "C"}:
            errors.append("confirmatory arms must be exactly A, B, and C")
        if arms["A"].get("current_product_mode") != "repeated_single_shot":
            errors.append("arm A must map to current product mode repeated_single_shot")
        if arms["B"].get("current_product_mode") != "iterative_no_checklist":
            errors.append("arm B must map to current product mode iterative_no_checklist")
        if arms["C"].get("current_product_mode") != "gym_with_checklist":
            errors.append("arm C must map to current product mode gym_with_checklist")
        if arms["D"].get("current_product_mode") != "fixed_transitions":
            errors.append("arm D must map to current product mode fixed_transitions")
        if arms["B"].get("checklist_feedback") is not False:
            errors.append("arm B must disable checklist feedback")
        if arms["C"].get("checklist_feedback") is not True:
            errors.append("arm C must enable checklist feedback")

    matrix = data.get("matrix") or {}
    models = matrix.get("models") or []
    datasets = matrix.get("datasets") or []
    confirmatory_arms = matrix.get("confirmatory_arms") or []
    replicates = matrix.get("replicates_per_cell")
    if len(models) != 2:
        errors.append("matrix must contain exactly two model slots")
    if len(datasets) != 4:
        errors.append("matrix must contain exactly four dataset slots")
    if confirmatory_arms != ["A", "B", "C"]:
        errors.append("matrix.confirmatory_arms must be ordered A, B, C")
    if not isinstance(replicates, int) or replicates < 1:
        errors.append("replicates_per_cell must be a positive integer")
    elif models and datasets and confirmatory_arms:
        expected_runs = len(models) * len(datasets) * len(confirmatory_arms) * replicates
        if matrix.get("planned_confirmatory_runs") != expected_runs:
            errors.append(
                "planned_confirmatory_runs must equal models * datasets * arms * replicates "
                f"({expected_runs})"
            )

    budget = data.get("budget") or {}
    for key in (
        "total_token_limit",
        "max_output_tokens_per_call",
        "max_llm_calls",
        "max_code_executions",
        "max_tool_calls",
        "wall_clock_limit_seconds",
    ):
        value = budget.get(key)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"budget.{key} must be a positive integer")
    if budget.get("max_output_tokens_per_call", 0) > budget.get("total_token_limit", 0):
        errors.append("per-call output cap cannot exceed the global token limit")
    if budget.get("pre_call_enforcement_required") is not True:
        errors.append("budget must require pre-call enforcement")
    if budget.get("stop_before_overshoot") is not True:
        errors.append("budget must stop before overshoot")

    submission = data.get("submission_policy") or {}
    if submission.get("host_side_autofit") is not False:
        errors.append("confirmatory submission policy must disable host-side autofit")
    if submission.get("hidden_score_can_drive_selection") is not False:
        errors.append("hidden score must not drive selection")
    if submission.get("max_hidden_evaluations_per_agent_outcome") != 1:
        errors.append("hidden evaluation must be attempted at most once per agent outcome")
    if submission.get("feedback_after_hidden_evaluation_can_change_outcome") is not False:
        errors.append("hidden evaluation cannot create an outcome-changing feedback loop")

    analysis = data.get("analysis") or {}
    primary = analysis.get("primary_comparison") or {}
    secondary = analysis.get("key_secondary_comparison") or {}
    if (primary.get("treatment"), primary.get("control")) != ("B", "A"):
        errors.append("primary comparison must be B versus A")
    if (secondary.get("treatment"), secondary.get("control")) != ("C", "B"):
        errors.append("key secondary comparison must be C versus B")

    failure_policy = data.get("failure_policy") or {}
    categories = set(failure_policy.get("categories") or [])
    if categories != EXPECTED_FAILURE_CATEGORIES:
        errors.append("failure-policy categories do not match the protocol taxonomy")
    if failure_policy.get("agent_failure_rerun") != "forbidden":
        errors.append("automatic agent-failure reruns must be forbidden")
    if failure_policy.get("overwrite_attempt_artifacts") is not False:
        errors.append("attempt artifacts must never be overwritten")

    checklist = data.get("checklist_validation") or {}
    if checklist.get("mandatory_detector_items") != 12:
        errors.append("checklist validation must target the current 12 mandatory items")
    if checklist.get("independent_annotators") != 2:
        errors.append("checklist validation must require two independent annotators")

    todos = data.get("todos") or []
    seen_todos: set[str] = set()
    open_freeze_blockers = 0
    for index, todo in enumerate(todos):
        if not isinstance(todo, dict):
            errors.append(f"todos[{index}] must be a mapping")
            continue
        todo_id = str(todo.get("id") or "")
        if not todo_id:
            errors.append(f"todos[{index}] is missing id")
        elif todo_id in seen_todos:
            errors.append(f"duplicate TODO id: {todo_id}")
        seen_todos.add(todo_id)
        for required in ("decision", "owner", "deadline", "status", "freeze_blocking"):
            if required not in todo:
                errors.append(f"TODO {todo_id or index} is missing {required}")
        if todo.get("freeze_blocking") is True and todo.get("status") != "resolved":
            open_freeze_blockers += 1

    freeze = data.get("freeze") or {}
    if freeze.get("allowed") is True and open_freeze_blockers:
        errors.append("freeze.allowed cannot be true while freeze-blocking TODOs are open")
    if data.get("status") == "frozen" and freeze.get("allowed") is not True:
        errors.append("a frozen protocol must set freeze.allowed=true")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate AutoVibe Gym Paper V1 protocol YAML.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)

    try:
        protocol = load_protocol(args.path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"INVALID: {exc}")
        return 1

    errors = validate_protocol(protocol)
    if errors:
        print("INVALID protocol:")
        for error in errors:
            print(f"- {error}")
        return 1

    matrix = protocol["matrix"]
    blockers = sum(
        1
        for todo in protocol["todos"]
        if todo.get("freeze_blocking") is True and todo.get("status") != "resolved"
    )
    print(
        "VALID protocol structure: "
        f"{matrix['planned_confirmatory_runs']} planned confirmatory runs; "
        f"{blockers} open freeze blocker(s); status={protocol['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

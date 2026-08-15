import json
from pathlib import Path

import yaml

from experiments import run_multishot
from gym.agent import SYSTEM_PROMPT, THOUGHTS_DISABLED_PROMPT
from gym.notebook_env import NotebookGymEnv
from research.planner import load_plan
from research.run_artifacts import canonical_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "research" / "protocols"


def _components():
    return json.loads(
        (PROTOCOLS / "confirmatory_components_v1.json").read_text(encoding="utf-8")
    )


def _gym_prompt(mode):
    return {
        "system": SYSTEM_PROMPT,
        "thoughts_policy": THOUGHTS_DISABLED_PROMPT,
        "episode_mode": mode,
        "protocol_version": NotebookGymEnv.protocol_version,
        "checklist_version": NotebookGymEnv.checklist_version,
        "feedback_policy_version": NotebookGymEnv.feedback_policy_version,
    }


def test_component_hashes_match_execution_commit_runtime_contracts():
    components = _components()
    a_prompt = {
        "system": run_multishot.SYSTEM_PROMPT,
        "task": run_multishot.TASK_PROMPT_TEMPLATE,
        "attempt_feedback": run_multishot.ATTEMPT_FEEDBACK_TEMPLATE,
    }
    prompts = {
        "A": a_prompt,
        "B": _gym_prompt("iterative_no_checklist"),
        "C": _gym_prompt("gym_with_checklist"),
    }

    assert canonical_hash(components["budget_policy"]["payload"]) == components[
        "budget_policy"
    ]["hash"]
    assert canonical_hash(components["decoding_config"]["payload"]) == components[
        "decoding_config"
    ]["hash"]
    for arm in ("A", "B", "C"):
        item = components["arms"][arm]
        assert canonical_hash(prompts[arm]) == item["prompt_template_hash"]
        assert canonical_hash(item["execution_policy"]) == item[
            "execution_policy_hash"
        ]


def test_freeze_record_binds_complete_result_blind_plan():
    record = json.loads(
        (PROTOCOLS / "freeze_record_v1.json").read_text(encoding="utf-8")
    )
    plan = load_plan(PROTOCOLS / "confirmatory_plan_v1.json")
    config = yaml.safe_load(
        (PROTOCOLS / "confirmatory_plan_config_v1.yaml").read_text(encoding="utf-8")
    )
    protocol = yaml.safe_load(
        (PROTOCOLS / "protocol_v1.yaml").read_text(encoding="utf-8")
    )
    components = _components()

    assert record["confirmatory_outcomes_visible_at_freeze"] is False
    assert plan["plan_id"] == record["plan_id"]
    assert plan["plan_hash"] == record["plan_hash"]
    assert plan["protocol_hash"] == record["protocol_hash"]
    assert plan["protocol_hash"] == canonical_hash(protocol)
    assert plan["config_hash"] == record["normalized_plan_config_hash"]
    assert len(plan["conditions"]) == record["planned_conditions"] == 120
    assert {
        item["condition"]["git_commit"] for item in plan["conditions"]
    } == {record["execution_git_commit"]}
    assert {
        arm: sum(item["condition"]["arm"] == arm for item in plan["conditions"])
        for arm in ("A", "B", "C")
    } == {"A": 40, "B": 40, "C": 40}
    assert config["git_commit"] == components["execution_git_commit"]
    assert config["budget_policy_hash"] == components["budget_policy"]["hash"]
    assert {
        item["decoding_config_hash"] for item in config["matrix"]["models"]
    } == {components["decoding_config"]["hash"]}
    for planned in config["matrix"]["arms"]:
        component = components["arms"][planned["arm"]]
        assert planned["product_mode"] == component["product_mode"]
        assert planned["prompt_template_hash"] == component["prompt_template_hash"]
        assert planned["execution_policy_hash"] == component[
            "execution_policy_hash"
        ]

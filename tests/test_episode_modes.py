from gym.modes import GYM_WITH_CHECKLIST, ITERATIVE_NO_CHECKLIST, resolve_episode_mode


def test_iterative_no_checklist_and_gym_modes_only_differ_by_checklist_feedback():
    no_checklist = resolve_episode_mode("iterative_no_checklist")
    gym_mode = resolve_episode_mode("gym_with_checklist")

    assert no_checklist.notebook_enabled == gym_mode.notebook_enabled
    assert no_checklist.runtime_feedback_enabled == gym_mode.runtime_feedback_enabled
    assert no_checklist.contract_feedback_enabled == gym_mode.contract_feedback_enabled
    assert no_checklist.checklist_feedback_enabled is False
    assert gym_mode.checklist_feedback_enabled is True


def test_mode_constants_resolve_to_themselves():
    assert resolve_episode_mode(ITERATIVE_NO_CHECKLIST) is ITERATIVE_NO_CHECKLIST
    assert resolve_episode_mode(None) is GYM_WITH_CHECKLIST

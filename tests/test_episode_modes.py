from gym.modes import DIRECTIVE_GYM, FREE_GYM, resolve_episode_mode


def test_free_and_directive_modes_only_differ_by_checklist_feedback():
    no_checklist = resolve_episode_mode("free_gym")
    directive_mode = resolve_episode_mode("directive_gym")

    assert no_checklist.notebook_enabled == directive_mode.notebook_enabled
    assert no_checklist.runtime_feedback_enabled == directive_mode.runtime_feedback_enabled
    assert no_checklist.contract_feedback_enabled == directive_mode.contract_feedback_enabled
    assert no_checklist.checklist_feedback_enabled is False
    assert directive_mode.checklist_feedback_enabled is True


def test_mode_constants_resolve_to_themselves():
    assert resolve_episode_mode(FREE_GYM) is FREE_GYM
    assert resolve_episode_mode(None) is DIRECTIVE_GYM

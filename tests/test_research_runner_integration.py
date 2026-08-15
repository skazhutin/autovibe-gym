import argparse

from research.runner_integration import start_research_run


def test_confirmatory_arm_override_is_bound_into_condition(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "research.runner_integration.git_provenance",
        lambda _repo: ("a" * 40, False),
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    args = argparse.Namespace(
        research_run_dir=str(tmp_path),
        research_experiment_id="paper-v1-confirmatory",
        research_replicate_index=2,
        research_arm="A",
        research_model_version="service-snapshot-2026-08-15",
        research_split_id="frozen-split-v1",
        research_condition_id=None,
        research_run_id="run_fixture",
        research_rerun_of=None,
        research_rerun_reason=None,
    )

    recorder = start_research_run(
        args,
        arm="repeated_single_shot",
        dataset_id="fixture",
        dataset_hash="b" * 64,
        split_seed=42,
        default_split_id="fallback-split",
        model_id="deepseek-v4-flash",
        prompt_template={"template": "fixture"},
        decoding_config={"max_tokens": 4096, "temperature": 0.4},
        budget_policy={"total_token_limit": 256000},
        execution_policy={"backend": "docker"},
    )
    try:
        assert recorder is not None
        assert recorder.manifest["condition"]["arm"] == "A"
        assert recorder.manifest["condition"]["split_id"] == "frozen-split-v1"
    finally:
        if recorder is not None:
            recorder.finalize({"final_status": "fixture", "valid_submit": False})

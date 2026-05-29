import pandas as pd

from gym.agent import GymAgent
from gym.env import GymEnv
from gym.llm import LLMResponse
from gym.protocol import Action


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, system, messages, model, max_tokens):
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "model": model,
                "max_tokens": max_tokens,
            }
        )
        response = self.responses.pop(0)
        return LLMResponse(text=response, input_tokens=3, output_tokens=5)


def _make_env(max_steps=3):
    train = pd.DataFrame({"x": [0, 1, 2, 3], "target": [0, 1, 0, 1]})
    val = pd.DataFrame({"x": [4, 5], "target": [0, 1]})
    test = pd.DataFrame({"x": [6, 7], "target": [0, 0]})
    return GymEnv(
        train=train,
        val=val,
        test=test,
        target_col="target",
        metric_fn=lambda y, p: sum(int(a == b) for a, b in zip(y, p)) / len(y),
        metric_name="accuracy",
        max_steps=max_steps,
        executor_backend="subprocess",
    )


def _train_constant_model_action():
    return (
        '{"type": "code", "code": "'
        "from sklearn.dummy import DummyClassifier\\n"
        "X_train = train_df.drop(columns=[target_col])\\n"
        "y_train = train_df[target_col]\\n"
        "model = DummyClassifier(strategy='constant', constant=0)\\n"
        "model.fit(X_train, y_train)"
        '"}'
    )


def test_agent_runs_code_then_submit_and_counts_tokens():
    client = ScriptedClient(
        [
            _train_constant_model_action(),
            '{"type": "submit", "model_var": "model"}',
        ]
    )
    agent = GymAgent(env=_make_env(), model="fake-model", max_tokens=123, client=client)

    summary = agent.run()

    assert summary["submitted"] is True
    assert summary["test_metric"] == 1.0
    assert summary["input_tokens"] == 6
    assert summary["output_tokens"] == 10
    assert client.calls[0]["model"] == "fake-model"
    assert client.calls[0]["max_tokens"] == 123


def test_agent_reprompts_when_llm_returns_invalid_json_action():
    client = ScriptedClient(
        [
            '{"type": "dance"}',
            _train_constant_model_action(),
            '{"type": "submit", "model_var": "model"}',
        ]
    )
    agent = GymAgent(env=_make_env(), client=client)

    summary = agent.run()

    assert summary["submitted"] is True
    assert "Could not parse your action" in client.calls[1]["messages"][-1]["content"]


def test_agent_forces_submit_after_budget_exhaustion_when_model_exists():
    client = ScriptedClient([_train_constant_model_action()])
    agent = GymAgent(env=_make_env(max_steps=1), client=client)

    summary = agent.run()

    assert summary["forced_submit"] is True
    assert summary["submitted"] is True
    assert summary["test_metric"] == 1.0


def test_agent_reports_max_agent_turns_when_client_never_submits_or_builds_model():
    client = ScriptedClient(['{"type": "code", "code": "print(1)"}'] * 11)
    agent = GymAgent(env=_make_env(max_steps=3), client=client)

    summary = agent.run()

    assert summary["forced_submit"] is True
    assert summary["submitted"] is False


def test_forced_submit_returns_none_without_model():
    agent = GymAgent(env=_make_env(), client=ScriptedClient([]))
    agent.env.reset()

    assert agent._try_forced_submit() is None


def test_forced_submit_prefers_best_model_over_model():
    env = _make_env()
    env.reset()
    env.step(Action.code_action(
        "from sklearn.dummy import DummyClassifier\n"
        "X_train = train_df.drop(columns=[target_col])\n"
        "y_train = train_df[target_col]\n"
        "model = DummyClassifier(strategy='constant', constant=1)\n"
        "model.fit(X_train, y_train)\n"
        "best_model = DummyClassifier(strategy='constant', constant=0)\n"
        "best_model.fit(X_train, y_train)"
    ))
    agent = GymAgent(env=env, client=ScriptedClient([]))

    observation = agent._try_forced_submit()

    assert observation.model_var == "best_model"
    assert observation.test_metric == 1.0

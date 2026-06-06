import unittest

import pandas as pd

from gym import Action, GymEnv


def accuracy(y_true, y_pred):
    pairs = zip(list(y_true), list(y_pred))
    return sum(int(actual == pred) for actual, pred in pairs) / len(y_true)


class EnvProtocolTests(unittest.TestCase):
    def make_env(self) -> GymEnv:
        train = pd.DataFrame(
            {
                "feature": [0, 1, 2, 3],
                "target": [0, 1, 0, 1],
            }
        )
        val = pd.DataFrame(
            {
                "feature": [4, 5],
                "target": [0, 1],
            }
        )
        test = pd.DataFrame(
            {
                "feature": [6, 7],
                "target": [0, 0],
            }
        )
        return GymEnv(
            train=train,
            val=val,
            test=test,
            target_col="target",
            metric_fn=accuracy,
            metric_name="accuracy",
            max_steps=3,
            executor_backend="subprocess",
        )

    def test_action_json_parsing(self):
        action = Action.from_llm_response(
            '{"type": "code", "stage": "data_schema_inspection", "code": "print(train_df.shape)"}'
        )

        self.assertEqual(action.type, "code")
        self.assertEqual(action.stage, "data_schema_inspection")
        self.assertEqual(action.code, "print(train_df.shape)")

    def test_workspace_is_initialized_and_persists_between_steps(self):
        env = self.make_env()
        env.reset()

        self.assertIn("train_df", env.state.namespace)
        self.assertIn("val_df", env.state.namespace)
        self.assertIn("target_col", env.state.namespace)
        self.assertNotIn("test_df", env.state.namespace)

        first = env.step({"type": "code", "stage": "feature_pipeline_building", "code": "value = 41"})
        second = env.step(Action.code_action("print(value + 1)"))

        self.assertEqual(first.step, 1)
        self.assertEqual(second.step, 2)
        self.assertIn("42", second.stdout)

    def test_submit_action_evaluates_named_model_once(self):
        env = self.make_env()
        env.reset()

        env.step(
            Action.code_action(
                """
from sklearn.dummy import DummyClassifier

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
best_model = DummyClassifier(strategy="constant", constant=0)
best_model.fit(X_train, y_train)
""".strip()
            )
        )
        observation = env.step({"type": "submit", "stage": "submission", "model_var": "best_model"})

        self.assertTrue(observation.submitted)
        self.assertTrue(observation.done)
        self.assertEqual(observation.test_metric, 1.0)
        self.assertEqual(env.state.step, 1)


if __name__ == "__main__":
    unittest.main()

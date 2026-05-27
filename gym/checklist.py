import re
from dataclasses import dataclass, field


@dataclass
class CheckItem:
    key: str
    passed: bool = False
    hint: str = ""  # shown to LLM when not yet passed


HINTS = {
    "eda": "Have you explored the dataset before modelling? (shape, dtypes, value counts)",
    "missing_values": "Some columns may have gaps — it might be worth addressing them.",
    "duplicates": "Duplicate rows can silently bias training — worth a quick check.",
    "target_leak": "Make sure no feature is a direct proxy of the target column.",
    "train_val_split": "It helps to evaluate on a held-out validation set during development.",
    "feature_engineering": "Raw features often aren't the best representation — any transformations tried?",
    "model_selection": "Have you compared more than one model type?",
    "hyperparameter_tuning": "Default hyperparameters are rarely optimal — even a small search can help.",
}


class Checklist:
    def __init__(self, target_col: str):
        self.target_col = target_col
        self.items: dict[str, CheckItem] = {
            key: CheckItem(key=key, hint=hint) for key, hint in HINTS.items()
        }

    def evaluate(
        self,
        code: str,
        stdout: str,
        namespace: dict,
        history: list,
    ) -> list[str]:
        """Update checklist state and return hints for items still not covered."""
        combined = (code + "\n" + stdout).lower()

        self._check_eda(combined, namespace)
        self._check_missing(combined, namespace)
        self._check_duplicates(combined)
        self._check_target_leak(code, namespace)
        self._check_train_val_split(combined, namespace)
        self._check_feature_engineering(combined)
        self._check_model_selection(history)
        self._check_hyperparameter_tuning(combined)

        return self._pending_hints()

    # --- individual checks ---

    def _check_eda(self, combined: str, ns: dict) -> None:
        if any(kw in combined for kw in ["describe(", ".info(", ".head(", "value_counts", ".shape"]):
            self.items["eda"].passed = True

    def _check_missing(self, combined: str, ns: dict) -> None:
        if any(kw in combined for kw in ["isnull", "isna", "fillna", "dropna", "impute"]):
            self.items["missing_values"].passed = True

    def _check_duplicates(self, combined: str) -> None:
        if any(kw in combined for kw in ["drop_duplicates", "duplicated()"]):
            self.items["duplicates"].passed = True

    def _check_target_leak(self, code: str, ns: dict) -> None:
        target = self.target_col.lower()
        # Pass if the agent explicitly drops the target from features
        if re.search(rf"drop\s*\(.*['\"]?{re.escape(target)}['\"]?", code.lower()):
            self.items["target_leak"].passed = True
        # Also pass if X and y are defined separately
        if re.search(r"\bX\b.*=.*drop|X_train|X_val", code):
            self.items["target_leak"].passed = True

    def _check_train_val_split(self, combined: str, ns: dict) -> None:
        if any(kw in combined for kw in ["val_df", "train_test_split", "validation"]):
            self.items["train_val_split"].passed = True

    def _check_feature_engineering(self, combined: str) -> None:
        keywords = [
            "log(", "sqrt(", "**2", "pd.get_dummies", "labelencoder",
            "onehotencoder", "standardscaler", "minmaxscaler",
            "polynomial", "interaction", "astype(", "map(",
        ]
        if any(kw in combined for kw in keywords):
            self.items["feature_engineering"].passed = True

    def _check_model_selection(self, history: list) -> None:
        model_keywords = [
            "logisticregression", "randomforest", "gradientboosting",
            "xgbclassifier", "xgbregressor", "lgbm", "svc", "decisiontree",
            "linearregression", "ridge", "lasso", "kneighbors",
        ]
        seen = set()
        for step in history:
            code_lower = step.code.lower()
            for kw in model_keywords:
                if kw in code_lower:
                    seen.add(kw)
        if len(seen) >= 2:
            self.items["model_selection"].passed = True

    def _check_hyperparameter_tuning(self, combined: str) -> None:
        if any(kw in combined for kw in ["gridsearchcv", "randomizedsearchcv", "optuna", "n_estimators=", "max_depth="]):
            self.items["hyperparameter_tuning"].passed = True

    # --- helpers ---

    def _pending_hints(self) -> list[str]:
        return [item.hint for item in self.items.values() if not item.passed]

    def coverage(self) -> float:
        passed = sum(1 for item in self.items.values() if item.passed)
        return round(passed / len(self.items), 2)

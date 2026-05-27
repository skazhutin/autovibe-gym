from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Workspace:
    """
    Mutable Python workspace visible to the LLM code.

    The hidden test split is intentionally not part of this namespace.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    target_col: str
    extra_globals: dict[str, Any] = field(default_factory=dict)
    namespace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.namespace:
            self.reset()

    def reset(self) -> None:
        self.namespace = self._initial_namespace()

    def update_namespace(self, namespace: dict[str, Any]) -> None:
        self.namespace = {key: value for key, value in namespace.items() if not key.startswith("_")}

    def get(self, name: str) -> Any:
        return self.namespace.get(name)

    def first_existing(self, names: list[str]) -> tuple[str | None, Any | None]:
        for name in names:
            value = self.namespace.get(name)
            if value is not None:
                return name, value
        return None, None

    def visible_symbols(self) -> list[str]:
        return sorted(key for key in self.namespace if not key.startswith("_"))

    def _initial_namespace(self) -> dict[str, Any]:
        namespace = {
            "train_df": self.train.copy(),
            "val_df": self.val.copy(),
            "target_col": self.target_col,
            "pd": pd,
            "np": np,
        }
        namespace.update(self.extra_globals)
        return namespace

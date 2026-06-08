from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FeedbackChannel = Literal["runtime", "contract", "checklist", "terminal"]
FeedbackSeverity = Literal["info", "warning", "blocker"]


@dataclass(frozen=True)
class FeedbackItem:
    channel: FeedbackChannel
    key: str
    message: str
    severity: FeedbackSeverity = "info"
    visible_to_agent: bool = True
    cell_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "key": self.key,
            "message": self.message,
            "severity": self.severity,
            "visible_to_agent": self.visible_to_agent,
            "cell_id": self.cell_id,
        }


@dataclass(frozen=True)
class FeedbackPolicy:
    max_checklist_hints_per_execution: int = 1
    hint_cooldown_executions: int = 2
    show_private_coverage_to_agent: bool = False
    do_not_emit_new_checklist_hint_when_runtime_error_present: bool = True
    do_not_emit_new_checklist_hint_when_contract_blocker_present: bool = True


GENERIC_CHECKLIST_HINTS: dict[str, str] = {
    "task_understanding": (
        "Before modelling, it is useful to record the target, task type, and "
        "quality metric you are optimizing."
    ),
    "schema_review": (
        "Before modelling, inspect the table shape, columns, and data types."
    ),
    "target_distribution_review": (
        "Check the target distribution or summary statistics before choosing a model."
    ),
    "missing_values_audit": (
        "Check whether features contain missing values and whether the final "
        "solution must handle them."
    ),
    "categorical_features_audit": (
        "Check which feature types are present and whether any require special "
        "handling before training."
    ),
    "duplicates_audit": (
        "Before making the model more complex, check whether duplicate rows may "
        "distort evaluation."
    ),
    "suspicious_columns_audit": (
        "Inspect features that may be identifiers, nearly unique, or unavailable "
        "during real inference."
    ),
    "target_exclusion": (
        "Make sure the target column is not included in the model features."
    ),
    "baseline_candidate_created": (
        "Before complex tuning, it is safer to create one simple candidate that can already validate and submit on raw rows."
    ),
    "reproducible_solution": (
        "The current solution should reproduce after a clean kernel restart; run "
        "the full notebook before final submission."
    ),
    "validation_evaluated": (
        "Once a candidate exists, use the environment validation action to get an "
        "official validation score."
    ),
    "submit_ready_artifact": (
        "Before final submission, make sure the selected candidate can predict raw "
        "validation features."
    ),
    "baseline_first": (
        "Before complex tuning, it is safer to create one simple candidate that can already validate and submit on raw rows."
    ),
    "raw_row_inference_ready": (
        "The final candidate should accept raw validation/test rows directly in predict()."
    ),
    "derived_features_inside_pipeline": (
        "If you derive new columns during training, make sure the final candidate can create the same columns inside predict() when it receives raw validation or hidden-test rows."
    ),
    "serialization_reproducibility": (
        "Before final submission, check that the candidate can be serialized and reloaded without relying on live-kernel-only state."
    ),
    "finalization_planning": (
        "Keep enough budget to validate or finalize a raw-row-ready candidate before the episode ends."
    ),
    "high_cardinality_handling": (
        "High-cardinality columns should be handled deliberately; avoid relying on identifiers that may not generalize."
    ),
    "unseen_categories_handling": (
        "Categorical encoders should handle categories that appear in validation/test but were not present during fitting."
    ),
    "candidate_validation_before_submit": (
        "Use check_candidate or validate before submit so raw-row and serialization failures are caught early."
    ),
}


MANDATORY_CHECKS: tuple[str, ...] = (
    "task_understanding",
    "schema_review",
    "target_distribution_review",
    "missing_values_audit",
    "categorical_features_audit",
    "duplicates_audit",
    "suspicious_columns_audit",
    "target_exclusion",
    "baseline_candidate_created",
    "validation_evaluated",
    "reproducible_solution",
    "submit_ready_artifact",
)

GUIDANCE_ONLY_CHECKS: tuple[str, ...] = (
    "baseline_first",
    "raw_row_inference_ready",
    "derived_features_inside_pipeline",
    "serialization_reproducibility",
    "finalization_planning",
    "high_cardinality_handling",
    "unseen_categories_handling",
    "candidate_validation_before_submit",
)


OPTIONAL_PROCESS_TAGS: tuple[str, ...] = (
    "compared_multiple_models",
    "attempted_controlled_tuning",
    "attempted_feature_engineering",
    "created_visualizations",
    "saved_model_artifact",
)


@dataclass
class ChecklistEvidence:
    key: str
    cell_id: str | None
    reason: str
    step: int

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "cell_id": self.cell_id,
            "reason": self.reason,
            "step": self.step,
        }


@dataclass
class NotebookChecklist:
    """Generic hidden checklist with selective, non-dataset-specific hints."""

    target_col: str
    policy: FeedbackPolicy = field(default_factory=FeedbackPolicy)
    covered: set[str] = field(default_factory=set)
    optional_tags: set[str] = field(default_factory=set)
    guidance_shown: set[str] = field(default_factory=set)
    evidence: list[ChecklistEvidence] = field(default_factory=list)
    executions_since_hint: int = 999
    hints_shown_total: int = 0
    process_guidance_hints_shown_total: int = 0

    def record_execution(
        self,
        *,
        source: str,
        stdout: str,
        cell_id: str | None,
        step: int,
        execution_success: bool,
        has_runtime_error: bool = False,
        has_contract_blocker: bool = False,
    ) -> list[FeedbackItem]:
        self.executions_since_hint += 1

        if execution_success and (source.strip() or stdout.strip()):
            self._record_behavioral_evidence(source, stdout, cell_id, step)

        if (
            has_runtime_error
            and self.policy.do_not_emit_new_checklist_hint_when_runtime_error_present
        ):
            return []
        if (
            has_contract_blocker
            and self.policy.do_not_emit_new_checklist_hint_when_contract_blocker_present
        ):
            return []
        if self.executions_since_hint < self.policy.hint_cooldown_executions:
            return []

        hint = self._next_hint()
        if hint is None:
            return []

        self.executions_since_hint = 0
        self.hints_shown_total += 1
        if hint in GUIDANCE_ONLY_CHECKS:
            self.guidance_shown.add(hint)
            self.process_guidance_hints_shown_total += 1
        return [
            FeedbackItem(
                channel="checklist",
                key=hint,
                message=GENERIC_CHECKLIST_HINTS[hint],
                severity="info",
                visible_to_agent=True,
                cell_id=cell_id,
            )
        ]

    def record_structural(self, key: str, *, reason: str, step: int) -> None:
        if key not in MANDATORY_CHECKS:
            return
        self._cover(key, cell_id=None, reason=reason, step=step)

    def record_optional_tag(self, key: str) -> None:
        if key in OPTIONAL_PROCESS_TAGS:
            self.optional_tags.add(key)

    def coverage(self) -> float:
        return round(len(self.covered) / len(MANDATORY_CHECKS), 2)

    def to_private_dict(self) -> dict:
        return {
            "covered": sorted(self.covered),
            "coverage": self.coverage(),
            "optional_tags": sorted(self.optional_tags),
            "guidance_shown": sorted(self.guidance_shown),
            "evidence": [item.to_dict() for item in self.evidence],
            "hints_shown_total": self.hints_shown_total,
            "process_guidance_hints_shown_total": self.process_guidance_hints_shown_total,
        }

    def _record_behavioral_evidence(
        self,
        source: str,
        stdout: str,
        cell_id: str | None,
        step: int,
    ) -> None:
        combined = f"{source}\n{stdout}".lower()
        target = self.target_col.lower()

        if target in combined and any(kw in combined for kw in ["metric", "target_col", "target", "score", "task"]):
            self._cover("task_understanding", cell_id=cell_id, reason="target/task context output", step=step)
        if any(kw in combined for kw in ["shape", "dtypes", "columns", "info()"]):
            self._cover("schema_review", cell_id=cell_id, reason="schema output", step=step)
        if target in combined and any(kw in combined for kw in ["value_counts", "describe", "nunique", "mean", "std"]):
            self._cover("target_distribution_review", cell_id=cell_id, reason="target distribution output", step=step)
        if any(kw in combined for kw in ["isna", "isnull", "missing", "null"]):
            self._cover("missing_values_audit", cell_id=cell_id, reason="missing-value audit output", step=step)
        if any(kw in combined for kw in ["select_dtypes", "object", "category", "categorical", "cardinality", "nunique"]):
            self._cover("categorical_features_audit", cell_id=cell_id, reason="feature type/cardinality output", step=step)
        if any(kw in combined for kw in ["duplicated", "duplicate", "drop_duplicates"]):
            self._cover("duplicates_audit", cell_id=cell_id, reason="duplicate audit output", step=step)
        if any(kw in combined for kw in ["nunique", "unique", "cardinality", "identifier", "id-like", "leak"]):
            self._cover("suspicious_columns_audit", cell_id=cell_id, reason="suspicious-column audit output", step=step)
        if "drop" in combined and target in combined and any(kw in combined for kw in ["x_train", "x_val", "features", "drop(columns"]):
            self._cover("target_exclusion", cell_id=cell_id, reason="target excluded from features", step=step)

        if any(kw in combined for kw in ["randomforest", "logisticregression", "gradientboosting", "xgb", "lgbm", "svc"]):
            self.record_optional_tag("compared_multiple_models")
        if any(kw in combined for kw in ["param_grid", "randomizedsearchcv", "gridsearchcv", "max_depth", "n_estimators"]):
            self.record_optional_tag("attempted_controlled_tuning")
        if any(kw in combined for kw in ["feature", "transform", "encoder", "scaler", "impute"]):
            self.record_optional_tag("attempted_feature_engineering")
        if any(kw in combined for kw in ["plt.", "plot(", "hist(", "scatter("]):
            self.record_optional_tag("created_visualizations")
        if any(kw in combined for kw in ["joblib.dump", "pickle.dump", ".pkl", ".joblib"]):
            self.record_optional_tag("saved_model_artifact")

    def _cover(self, key: str, *, cell_id: str | None, reason: str, step: int) -> None:
        if key in self.covered:
            return
        self.covered.add(key)
        self.evidence.append(
            ChecklistEvidence(key=key, cell_id=cell_id, reason=reason, step=step)
        )

    def _next_hint(self) -> str | None:
        for key in MANDATORY_CHECKS:
            if key not in self.covered:
                return key
        for key in GUIDANCE_ONLY_CHECKS:
            if key not in self.guidance_shown:
                return key
        return None

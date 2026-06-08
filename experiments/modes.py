"""Shared product-mode metadata for experiment launchers and reporting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ALL_REQUESTED_MODE = "all"
BATCH_REQUESTED_MODE = "batch"


@dataclass(frozen=True)
class ProductMode:
    key: str
    display_name: str
    matrix_label: str
    dashboard_mode: str
    module: str
    experiment_type: str
    mode_order: int
    episode_mode: str | None = None


SINGLE_SHOT = ProductMode(
    key="single_shot",
    display_name="Single-shot",
    matrix_label="single-shot",
    dashboard_mode="single",
    module="experiments.run_baseline",
    experiment_type="baseline_single_shot",
    mode_order=1,
)

REPEATED_SINGLE_SHOT = ProductMode(
    key="repeated_single_shot",
    display_name="Repeated single-shot",
    matrix_label="repeated single-shot",
    dashboard_mode="repeated",
    module="experiments.run_multishot",
    experiment_type="repeated_single_shot",
    mode_order=2,
)

GYM_WITH_CHECKLIST = ProductMode(
    key="gym_with_checklist",
    display_name="Free gym",
    matrix_label="free gym",
    dashboard_mode="gym",
    module="experiments.run_gym",
    experiment_type="gym_with_checklist",
    mode_order=4,
    episode_mode="gym_with_checklist",
)

FIXED_TRANSITIONS = ProductMode(
    key="fixed_transitions",
    display_name="Directive gym",
    matrix_label="directive gym",
    dashboard_mode="fixed",
    module="experiments.run_fixed",
    experiment_type="fixed_transitions",
    mode_order=5,
)

ITERATIVE_NO_CHECKLIST = ProductMode(
    key="iterative_no_checklist",
    display_name="Iterative no checklist",
    matrix_label="iterative no-checklist",
    dashboard_mode="iterative",
    module="experiments.run_gym",
    experiment_type="iterative_no_checklist",
    mode_order=3,
    episode_mode="iterative_no_checklist",
)

# Product modes presented in README quickstart and the all-modes matrix.
ALL_PRODUCT_MODES: tuple[ProductMode, ...] = (
    SINGLE_SHOT,
    REPEATED_SINGLE_SHOT,
    ITERATIVE_NO_CHECKLIST,
    GYM_WITH_CHECKLIST,
    FIXED_TRANSITIONS,
)

RUNNABLE_MODES: tuple[ProductMode, ...] = (
    SINGLE_SHOT,
    REPEATED_SINGLE_SHOT,
    ITERATIVE_NO_CHECKLIST,
    GYM_WITH_CHECKLIST,
    FIXED_TRANSITIONS,
)

MODE_BY_KEY = {mode.key: mode for mode in RUNNABLE_MODES}
MODE_BY_EXPERIMENT_TYPE = {mode.experiment_type: mode for mode in RUNNABLE_MODES}
MODE_BY_DASHBOARD_MODE = {mode.dashboard_mode: mode for mode in RUNNABLE_MODES}

MODE_ALIASES = {
    "single": SINGLE_SHOT.key,
    "single-shot": SINGLE_SHOT.key,
    "baseline": SINGLE_SHOT.key,
    "baseline_single_shot": SINGLE_SHOT.key,
    "repeated": REPEATED_SINGLE_SHOT.key,
    "repeated-single-shot": REPEATED_SINGLE_SHOT.key,
    "multishot": REPEATED_SINGLE_SHOT.key,
    "iterative": ITERATIVE_NO_CHECKLIST.key,
    "iterative-no-checklist": ITERATIVE_NO_CHECKLIST.key,
    "gym": GYM_WITH_CHECKLIST.key,
    "gym-with-checklist": GYM_WITH_CHECKLIST.key,
    "flexible-gym": GYM_WITH_CHECKLIST.key,
    "fixed": FIXED_TRANSITIONS.key,
    "fixed-transitions": FIXED_TRANSITIONS.key,
}


def normalize_mode_key(mode: str) -> str:
    normalized = mode.strip().lower().replace(" ", "-")
    if normalized == ALL_REQUESTED_MODE:
        return ALL_REQUESTED_MODE
    normalized = MODE_ALIASES.get(normalized, normalized.replace("-", "_"))
    if normalized not in MODE_BY_KEY:
        valid = ", ".join([ALL_REQUESTED_MODE, *MODE_BY_KEY])
        raise ValueError(f"Unsupported run mode '{mode}'. Expected one of: {valid}")
    return normalized


def expand_requested_mode(mode: str) -> tuple[ProductMode, ...]:
    key = normalize_mode_key(mode)
    if key == ALL_REQUESTED_MODE:
        return ALL_PRODUCT_MODES
    return (MODE_BY_KEY[key],)


def add_mode_metadata_args(parser: Any) -> None:
    parser.add_argument("--requested-mode", default=None)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--product-mode", default=None)
    parser.add_argument("--mode-label", default=None)
    parser.add_argument("--mode-order", type=int, default=None)


def mode_metadata_params(args: Any, default_product_mode: str) -> dict[str, Any]:
    mode = MODE_BY_KEY.get(default_product_mode) or MODE_BY_EXPERIMENT_TYPE[default_product_mode]
    product_mode = args.product_mode or mode.key
    mode_label = args.mode_label or product_mode
    mode_order = args.mode_order if args.mode_order is not None else mode.mode_order
    return {
        "requested_mode": args.requested_mode or product_mode,
        "batch_id": args.batch_id or "",
        "product_mode": product_mode,
        "mode_label": mode_label,
        "mode_order": mode_order,
    }


def mode_metadata_cli_args(
    mode: ProductMode,
    *,
    requested_mode: str,
    batch_id: str | None,
) -> list[str]:
    args = [
        "--requested-mode",
        requested_mode,
        "--product-mode",
        mode.key,
        "--mode-label",
        mode.key,
        "--mode-order",
        str(mode.mode_order),
    ]
    if batch_id:
        args.extend(["--batch-id", batch_id])
    return args

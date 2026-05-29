"""Tests for generic Checklist wrapper."""
import pytest

from gym.checklist import Checklist


@pytest.fixture
def cl():
    return Checklist(target_col="quality")


def test_initial_coverage_zero(cl):
    assert cl.coverage() == 0.0


def test_behavioral_schema_check_requires_output(cl):
    cl.evaluate(code="train_df.shape", stdout="", namespace={}, history=[])
    assert not cl.items["schema_review"].passed

    cl.evaluate(code="print(train_df.shape)", stdout="(10, 3)", namespace={}, history=[])
    assert cl.items["schema_review"].passed


def test_missing_values_check_is_generic(cl):
    hints = cl.evaluate(
        code="print(train_df.isnull().sum())",
        stdout="x    0",
        namespace={},
        history=[],
    )

    assert cl.items["missing_values_audit"].passed
    assert len(hints) <= 1
    assert all("SimpleImputer" not in hint for hint in hints)


def test_target_distribution_and_exclusion_checks(cl):
    cl.evaluate(
        code="print(train_df['quality'].value_counts())",
        stdout="0    5\n1    5",
        namespace={},
        history=[],
    )
    cl.evaluate(
        code="X_train = train_df.drop(columns=['quality']); print(X_train.columns)",
        stdout="Index(['x'], dtype='object')",
        namespace={},
        history=[],
    )

    assert cl.items["target_distribution_review"].passed
    assert cl.items["target_exclusion"].passed


def test_duplicate_and_suspicious_column_checks(cl):
    cl.evaluate(
        code="print(train_df.duplicated().sum())",
        stdout="0",
        namespace={},
        history=[],
    )
    cl.evaluate(
        code="print(train_df.nunique())",
        stdout="id    100",
        namespace={},
        history=[],
    )

    assert cl.items["duplicates_audit"].passed
    assert cl.items["suspicious_columns_audit"].passed


def test_structural_checks_update_coverage(cl):
    for key in [
        "baseline_candidate_created",
        "validation_evaluated",
        "reproducible_solution",
        "submit_ready_artifact",
    ]:
        cl.record_structural(key, reason="unit")

    assert cl.items["validation_evaluated"].passed
    assert cl.coverage() > 0


def test_hints_are_selective_not_full_private_coverage(cl):
    hints = cl.evaluate(code="print('hello')", stdout="hello", namespace={}, history=[])

    assert len(hints) == 1
    assert "Coverage=" not in hints[0]
    assert "Missing checklist item" not in hints[0]

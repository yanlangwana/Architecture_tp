import pytest

from fraudlake.config import PSI_ALERT
from fraudlake.monitoring import psi, psi_status, retraining_decision


def test_psi_is_zero_for_identical_distributions():
    ref = {0: 100, 1: 200, 2: 300}
    assert psi(ref, {0: 10, 1: 20, 2: 30}) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_a_strong_shift():
    ref = {i: 100 for i in range(10)}
    cur = {**{i: 10 for i in range(5)}, **{i: 190 for i in range(5, 10)}}
    assert psi(ref, cur) > PSI_ALERT


def test_psi_handles_new_and_missing_categories():
    value = psi({"a": 50, "b": 50}, {"a": 50, "c": 50})
    assert value > PSI_ALERT and value < 10


@pytest.mark.parametrize("value,status", [(0.02, "stable"), (0.15, "à surveiller"), (0.4, "dérive")])
def test_psi_status(value, status):
    assert psi_status(value) == status


def test_retraining_decision():
    assert retraining_decision(0.05, 0.80, 0.70) == (False, "aucun déclencheur")
    trigger, why = retraining_decision(0.31, 0.80, 0.70)
    assert trigger and "dérive" in why
    trigger, why = retraining_decision(0.05, 0.60, 0.70, pr_auc=0.70, pr_auc_ref=0.89)
    assert trigger and "rappel" in why and "PR-AUC" in why

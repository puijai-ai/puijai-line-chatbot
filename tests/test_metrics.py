import math

import pytest

from puijai_risk.evaluation.kappa import cohens_kappa, interpret_kappa
from puijai_risk.evaluation.metrics import classification_report, confusion_matrix, erde, latency_cost


def test_confusion_and_report():
    golds = [0, 0, 1, 2, 2]
    preds = [0, 1, 1, 2, 0]
    assert confusion_matrix(golds, preds) == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]
    rep = classification_report(golds, preds)
    assert rep["CRITICAL"]["precision"] == 1.0
    assert rep["CRITICAL"]["recall"] == 0.5
    assert rep["critical_recall"] == 0.5
    assert rep["accuracy"] == pytest.approx(0.6)
    assert rep["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 2 / 3) / 3)


def test_accuracy_trap_on_imbalanced_data():
    """โมเดลที่ทาย NORMAL ทั้งหมด: Accuracy สูง แต่ Recall ของ CRITICAL = 0."""
    golds = [0] * 95 + [2] * 5
    preds = [0] * 100
    rep = classification_report(golds, preds)
    assert rep["accuracy"] == 0.95
    assert rep["critical_recall"] == 0.0
    assert rep["CRITICAL"]["f1"] == 0.0


def test_latency_cost():
    assert latency_cost(5, 5) == pytest.approx(0.5)
    assert latency_cost(1, 5) < 0.05
    assert latency_cost(1000, 5) == pytest.approx(1.0)


def test_erde():
    # TP ที่ k=5 (o=5) → 0.5, FP → c_fp = 0.5 (สัดส่วน positive)
    assert erde([True, False], [True, True], [5, 1], o=5) == pytest.approx(0.5)
    # FN → 1, TN → 0
    assert erde([True, False], [False, False], [3, 3], o=5) == pytest.approx(0.5)
    # ตัดสินเร็วดีกว่าตัดสินช้า
    assert erde([True], [True], [1], o=5) < erde([True], [True], [10], o=5)
    assert not math.isnan(erde([], [], []))


def test_cohens_kappa():
    assert cohens_kappa([0, 1, 2, 0], [0, 1, 2, 0]) == pytest.approx(1.0)
    assert cohens_kappa([0, 1, 2, 0], [0, 1, 2, 0], weighted=True) == pytest.approx(1.0)
    k = cohens_kappa([0, 1, 2, 1, 0], [0, 1, 1, 1, 0], weighted=True)
    assert 0.5 < k < 1.0
    assert "agreement" in interpret_kappa(k).lower()


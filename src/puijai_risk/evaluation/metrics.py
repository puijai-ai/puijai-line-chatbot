"""ตัววัดผล: Precision / Recall / F1 ต่อคลาส, Macro-F1, Confusion Matrix และ ERDE.

ไม่พึ่ง scikit-learn เพื่อให้รัน test ได้ทุกเครื่อง
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..schemas import LABELS, RiskLevel


def confusion_matrix(golds: Sequence[int], preds: Sequence[int], n_classes: int = len(LABELS)) -> list[list[int]]:
    """แถว = ค่าจริง, คอลัมน์ = ค่าที่ทาย."""
    m = [[0] * n_classes for _ in range(n_classes)]
    for g, p in zip(golds, preds):
        m[int(g)][int(p)] += 1
    return m


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def classification_report(golds: Sequence[int], preds: Sequence[int]) -> dict:
    if len(golds) != len(preds):
        raise ValueError("golds and preds must have the same length")
    cm = confusion_matrix(golds, preds)
    report: dict = {}
    f1s = []
    for lvl in LABELS:
        i = int(lvl)
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(LABELS))) - tp
        fn = sum(cm[i]) - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        f1s.append(f1)
        report[lvl.name] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(cm[i])}
    report["accuracy"] = _safe_div(sum(cm[i][i] for i in range(len(LABELS))), len(golds))
    report["macro_f1"] = sum(f1s) / len(f1s)
    report["critical_recall"] = report[RiskLevel.CRITICAL.name]["recall"]
    report["confusion_matrix"] = cm
    return report


def latency_cost(k: int, o: int) -> float:
    """lc_o(k) = 1 - 1/(1+e^(k-o)) — ยิ่งตัดสินช้า (k มาก) ยิ่งเสียค่าปรับ."""
    return 1.0 - 1.0 / (1.0 + math.exp(max(min(k - o, 50), -50)))


def erde(
    golds: Sequence[bool],
    decisions: Sequence[bool],
    delays: Sequence[int],
    o: int = 5,
    c_fp: float | None = None,
    c_fn: float = 1.0,
    c_tp: float = 1.0,
) -> float:
    """Early Risk Detection Error (Losada & Crestani, 2016). ค่ายิ่งต่ำยิ่งดี.

    golds     : บทสนทนานี้มีจุดเสี่ยงจริงหรือไม่
    decisions : ระบบแจ้งเตือนหรือไม่
    delays    : จำนวนข้อความ (k) ที่ระบบใช้ก่อนแจ้งเตือน
    c_fp      : ค่าปรับ false positive (ค่าเริ่มต้น = สัดส่วนผู้มีความเสี่ยงในชุดข้อมูล)
    """
    n = len(golds)
    if n == 0:
        return 0.0
    if c_fp is None:
        c_fp = sum(1 for g in golds if g) / n
    total = 0.0
    for g, d, k in zip(golds, decisions, delays):
        if d and not g:
            total += c_fp
        elif g and not d:
            total += c_fn
        elif g and d:
            total += c_tp * latency_cost(k, o)
    return total / n


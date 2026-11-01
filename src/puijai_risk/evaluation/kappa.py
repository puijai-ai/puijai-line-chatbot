"""คำนวณค่าความสอดคล้องในการติดป้ายข้อมูลระหว่างผู้ประเมิน 2 คน (Cohen's Kappa).

ใช้ในบทที่ 3 เพื่อพิสูจน์คุณภาพของชุดข้อมูลที่คนที่ A และคนที่ B ติดป้ายร่วมกัน:
    python -m puijai_risk.evaluation.kappa --annotator-a data/sample/sample_conversations.csv --annotator-b data/sample/sample_conversations.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ..data import load_conversations
from ..schemas import LABELS, RiskLevel
from .metrics import confusion_matrix


def cohens_kappa(labels_a: Sequence[int], labels_b: Sequence[int], weighted: bool = False) -> float:
    """คำนวณ Cohen's Kappa (รองรับทั้งแบบปกติ และ Quadratic Weighted สำหรับลำดับความเสี่ยง 0-1-2)."""
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have the same length")
    n = len(labels_a)
    if n == 0:
        return 0.0
    k = len(LABELS)
    cm = confusion_matrix(labels_a, labels_b, n_classes=k)
    row_totals = [sum(cm[i]) for i in range(k)]
    col_totals = [sum(cm[r][j] for r in range(k)) for j in range(k)]

    if not weighted:
        po = sum(cm[i][i] for i in range(k)) / n
        pe = sum(row_totals[i] * col_totals[i] for i in range(k)) / (n * n)
        return 1.0 if pe == 1.0 else (po - pe) / (1.0 - pe)

    # Quadratic weighted disagreement
    obs_dis = 0.0
    exp_dis = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            obs_dis += w * (cm[i][j] / n)
            exp_dis += w * ((row_totals[i] * col_totals[j]) / (n * n))
    return 1.0 if exp_dis == 0.0 else 1.0 - (obs_dis / exp_dis)


def interpret_kappa(kappa: float) -> str:
    if kappa >= 0.81:
        return "Almost perfect agreement (สอดคล้องกันดีเยี่ยม)"
    if kappa >= 0.61:
        return "Substantial agreement (สอดคล้องกันในระดับดี — ผ่านเกณฑ์วิจัย)"
    if kappa >= 0.41:
        return "Moderate agreement (สอดคล้องกันปานกลาง — ควรทบทวนคู่มือติดป้าย)"
    return "Fair/Poor agreement (ควรปรับเกณฑ์การติดป้ายใหม่)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Cohen's Kappa between two annotator CSV files")
    ap.add_argument("--annotator-a", required=True)
    ap.add_argument("--annotator-b", required=True)
    args = ap.parse_args()

    convs_a = load_conversations(args.annotator_a)
    convs_b = load_conversations(args.annotator_b)
    a_list: list[int] = []
    b_list: list[int] = []
    for cid in sorted(set(convs_a) & set(convs_b)):
        for ta, tb in zip(convs_a[cid], convs_b[cid]):
            a_list.append(int(ta.label))
            b_list.append(int(tb.label))

    k_unweighted = cohens_kappa(a_list, b_list, weighted=False)
    k_weighted = cohens_kappa(a_list, b_list, weighted=True)
    print(f"Compared {len(a_list)} turns across {len(set(convs_a) & set(convs_b))} conversations")
    print(f"Cohen's Kappa (Unweighted)         : {k_unweighted:.4f} -> {interpret_kappa(k_unweighted)}")
    print(f"Cohen's Kappa (Quadratic Weighted) : {k_weighted:.4f} -> {interpret_kappa(k_weighted)}")
    for lvl in LABELS:
        ca = sum(1 for x in a_list if x == int(lvl))
        cb = sum(1 for x in b_list if x == int(lvl))
        print(f"  {lvl.name:8s}: Annotator A = {ca:3d}, Annotator B = {cb:3d}")


if __name__ == "__main__":
    main()


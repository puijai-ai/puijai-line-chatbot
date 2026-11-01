"""การทดลองบทที่ 4: "ต้องใช้กี่คำ / กี่ข้อความ จึงจับจุดเสี่ยงได้ดีที่สุด"

Exp.1  แบ่งข้อความตามความยาว (<5, 5–20, >20 คำ) แล้ววัด Precision/Recall/F1 (N = 1 ข้อความ)
Exp.2  เปลี่ยนจำนวนข้อความสะสม N = 1, 3, 5 แล้ววัด Macro-F1, Recall(CRITICAL), ERDE_5, ERDE_50

ตัวอย่าง:
    python -m puijai_risk.evaluation.length_experiment --classifier keyword --data data/sample/sample_conversations.csv
    python -m puijai_risk.evaluation.length_experiment --classifier wangchanberta --model-path models/wangchanberta-risk --data data/processed/test.csv

หมายเหตุ: --data ควรเป็นชุดทดสอบที่ไม่เคยใช้เทรนโมเดล
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from ..classifiers import RiskClassifier, get_classifier
from ..data import Conversations, WindowSample, conversation_gold, load_conversations, make_window_samples
from ..schemas import RiskLevel
from ..windowing import Window, prefix_windows
from .metrics import classification_report, erde

LENGTH_BUCKETS = [("<5", 0, 4), ("5-20", 5, 20), (">20", 21, 10**9)]


def exp1_length_buckets(clf: RiskClassifier, convs: Conversations, max_tokens: int | None) -> list[dict]:
    samples = make_window_samples(convs, n_turns=1, max_tokens=max_tokens)
    preds = [r.level for r in clf.predict([s.text for s in samples])]
    rows = []
    for name, lo, hi in LENGTH_BUCKETS:
        idx = [i for i, s in enumerate(samples) if lo <= s.current_turn_tokens <= hi]
        if not idx:
            rows.append({"bucket": name, "n": 0})
            continue
        rep = classification_report([samples[i].label for i in idx], [preds[i] for i in idx])
        rows.append(_summary_row({"bucket": name, "n": len(idx)}, rep))
    return rows


def simulate_early_detection(
    clf: RiskClassifier,
    convs: Conversations,
    n_turns: int,
    max_tokens: int | None,
    positive_level: RiskLevel,
) -> tuple[list[bool], list[bool], list[int]]:
    """อ่านบทสนทนาทีละข้อความ หยุดเมื่อระบบแจ้งเตือนครั้งแรก → ได้ (gold, decision, delay)."""
    golds, decisions, delays = [], [], []
    all_windows, owners = [], []
    for cid, turns in convs.items():
        ws = prefix_windows([t.text for t in turns], n_turns=n_turns, max_tokens=max_tokens)
        all_windows.extend(ws)
        owners.extend([cid] * len(ws))
    results = clf.predict_windows(all_windows)

    by_conv: dict[str, list[RiskLevel]] = {}
    for cid, r in zip(owners, results):
        by_conv.setdefault(cid, []).append(r.level)

    for cid, turns in convs.items():
        levels = by_conv[cid]
        first = next((k for k, lvl in enumerate(levels, start=1) if lvl >= positive_level), None)
        golds.append(conversation_gold(turns) >= positive_level)
        decisions.append(first is not None)
        delays.append(first if first is not None else len(turns))
    return golds, decisions, delays


def exp2_context_window(
    clf: RiskClassifier,
    convs: Conversations,
    n_turns_list: list[int],
    max_tokens: int | None,
    positive_level: RiskLevel,
) -> list[dict]:
    rows = []
    for n in n_turns_list:
        samples = make_window_samples(convs, n_turns=n, max_tokens=max_tokens)
        t0 = time.perf_counter()
        results = clf.predict_windows([_as_window(s) for s in samples])
        elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / max(len(samples), 1)
        rep = classification_report([s.label for s in samples], [r.level for r in results])
        g, d, k = simulate_early_detection(clf, convs, n, max_tokens, positive_level)
        avg_tokens = sum(s.n_tokens for s in samples) / max(len(samples), 1)
        row = _summary_row({"n_turns": n, "n": len(samples), "avg_tokens": round(avg_tokens, 2)}, rep)
        row["erde_5"] = round(erde(g, d, k, o=5), 4)
        row["erde_50"] = round(erde(g, d, k, o=50), 4)
        row["latency_ms"] = round(elapsed_ms, 2)
        rows.append(row)
    return rows


def _as_window(sample: WindowSample) -> Window:
    return Window(text=sample.text, n_turns=sample.n_turns, n_tokens=sample.n_tokens)


def _summary_row(prefix: dict, rep: dict) -> dict:
    return {
        **prefix,
        "accuracy": round(rep["accuracy"], 4),
        "macro_f1": round(rep["macro_f1"], 4),
        "critical_precision": round(rep["CRITICAL"]["precision"], 4),
        "critical_recall": round(rep["CRITICAL"]["recall"], 4),
        "critical_f1": round(rep["CRITICAL"]["f1"], 4),
        "warning_recall": round(rep["WARNING"]["recall"], 4),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_exp2(rows: list[dict], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    xs = [r["n_turns"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("macro_f1", "critical_recall", "erde_5"):
        ax.plot(xs, [r[key] for r in rows], marker="o", label=key)
    ax.set_xlabel("N (messages in context window)")
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Length / context-window experiment")
    ap.add_argument("--classifier", default="keyword", choices=["keyword", "tfidf", "wangchanberta"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-turns", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--positive-level", default="CRITICAL", choices=["WARNING", "CRITICAL"])
    ap.add_argument("--critical-threshold", type=float, default=0.35)
    ap.add_argument("--warning-threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    clf = get_classifier(
        args.classifier,
        args.model_path,
        critical_threshold=args.critical_threshold,
        warning_threshold=args.warning_threshold,
    )
    convs = load_conversations(args.data)
    out = Path(args.out_dir)
    positive = RiskLevel[args.positive_level]

    rows1 = exp1_length_buckets(clf, convs, args.max_tokens)
    rows2 = exp2_context_window(clf, convs, args.n_turns, args.max_tokens, positive)
    p1 = out / f"length_experiment_{clf.name}_exp1.csv"
    p2 = out / f"length_experiment_{clf.name}_exp2.csv"
    write_csv(rows1, p1)
    write_csv(rows2, p2)

    print(f"[Exp.1] message length buckets -> {p1}")
    for r in rows1:
        print("  ", r)
    print(f"[Exp.2] context window N -> {p2}")
    for r in rows2:
        print("  ", r)
    png = out / f"length_experiment_{clf.name}_exp2.png"
    if plot_exp2(rows2, png):
        print(f"plot -> {png}")

    # บันทึกตารางสรุปเปรียบเทียบโมเดล (Exp.0 ที่ N=3 หรือค่ากลาง)
    target_row = next((r for r in rows2 if r["n_turns"] == 3), rows2[-1])
    summary_path = out / "model_comparison_summary.csv"
    existing: dict[str, dict] = {}
    if summary_path.exists():
        with open(summary_path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                existing[r["model"]] = r
    existing[clf.name] = {"model": clf.name, **target_row}
    write_csv(list(existing.values()), summary_path)
    print(f"[Exp.0] model comparison summary -> {summary_path}")


if __name__ == "__main__":
    main()

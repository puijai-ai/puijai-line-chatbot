"""Baseline 2: TF-IDF + Logistic Regression / Linear SVM (ต้องติดตั้ง requirements-ml.txt).

เทรน:
    python -m puijai_risk.classifiers.tfidf_svm --data data/processed/train.csv --out models/tfidf.joblib --n-turns 3
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..data import load_conversations, make_window_samples
from ..preprocessing import normalize_text, tokenize
from ..schemas import LABELS, RiskLevel
from .base import Proba, RiskClassifier


def _build_pipeline(algo: str):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=normalize_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    if algo == "svm":
        clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)
    else:
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


class TfidfClassifier(RiskClassifier):
    name = "tfidf"

    def __init__(self, pipeline, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline

    @classmethod
    def train(cls, texts: Sequence[str], labels: Sequence[int], algo: str = "lr", **kwargs) -> "TfidfClassifier":
        pipeline = _build_pipeline(algo)
        pipeline.fit(list(texts), [int(y) for y in labels])
        return cls(pipeline, **kwargs)

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "TfidfClassifier":
        import joblib

        return cls(joblib.load(path), **kwargs)

    def predict_proba(self, texts: Sequence[str]) -> list[Proba]:
        probs = self.pipeline.predict_proba(list(texts))
        classes = [RiskLevel(int(c)) for c in self.pipeline.classes_]
        out = []
        for row in probs:
            p = {lvl: 0.0 for lvl in LABELS}
            for lvl, v in zip(classes, row):
                p[lvl] = float(v)
            out.append(p)
        return out

    def explain(self, text: str) -> list[str]:
        """ดึงคำหรือวลี (1-2 grams) ในข้อความที่มีน้ำหนัก TF-IDF * ค่าสัมประสิทธิ์ของคลาสเสี่ยงสูงสุด."""
        tokens = set(tokenize(text))
        if not tokens:
            return []
        vec = self.pipeline.named_steps.get("tfidf")
        clf = self.pipeline.named_steps.get("clf")
        if vec is None or not hasattr(clf, "coef_"):
            return []
        x = vec.transform([text])
        feature_names = vec.get_feature_names_out()
        nonzero = x.nonzero()[1]
        if len(nonzero) == 0:
            return []
        # ใช้น้ำหนักของคลาส CRITICAL และ WARNING รวมกัน
        coefs = clf.coef_
        risk_weights = coefs[-1] + (coefs[-2] if coefs.shape[0] >= 3 else 0.0)
        scored = [
            (float(x[0, idx] * risk_weights[idx]), str(feature_names[idx]))
            for idx in nonzero
            if float(x[0, idx] * risk_weights[idx]) > 0
        ]
        scored.sort(reverse=True)
        return [name for _, name in scored[:5]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TF-IDF baseline")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/tfidf.joblib")
    ap.add_argument("--n-turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--algo", choices=["lr", "svm"], default="lr")
    args = ap.parse_args()

    samples = make_window_samples(load_conversations(args.data), args.n_turns, args.max_tokens)
    model = TfidfClassifier.train([s.text for s in samples], [s.label for s in samples], algo=args.algo)
    model.save(args.out)
    print(f"saved {args.out} ({len(samples)} samples)")


if __name__ == "__main__":
    main()


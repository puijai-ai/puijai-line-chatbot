"""Factory สำหรับเลือกตัวจำแนกจากชื่อ (ใช้ใน .env: RISK_MODEL=keyword|tfidf|wangchanberta)."""

from __future__ import annotations

from .base import RiskClassifier, decide_level
from .keyword import KeywordClassifier


def get_classifier(name: str, model_path: str | None = None, **kwargs) -> RiskClassifier:
    name = name.lower().strip()
    if name == "keyword":
        return KeywordClassifier(**kwargs)
    if name == "tfidf":
        from .tfidf_svm import TfidfClassifier

        if not model_path:
            raise ValueError("tfidf ต้องระบุ model_path (.joblib)")
        return TfidfClassifier.load(model_path, **kwargs)
    if name == "wangchanberta":
        from .wangchanberta import WangchanBertaClassifier

        if not model_path:
            raise ValueError("wangchanberta ต้องระบุ model_path (โฟลเดอร์โมเดล)")
        return WangchanBertaClassifier(model_path, **kwargs)
    raise ValueError(f"unknown classifier: {name}")


__all__ = ["RiskClassifier", "KeywordClassifier", "decide_level", "get_classifier"]


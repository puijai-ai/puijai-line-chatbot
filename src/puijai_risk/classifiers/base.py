"""Interface กลางของตัวจำแนกความเสี่ยง — ทั้ง A และ B ใช้ร่วมกัน."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..preprocessing import count_tokens
from ..schemas import RiskLevel, RiskResult
from ..windowing import Window

Proba = dict[RiskLevel, float]


def decide_level(probs: Proba, critical_threshold: float, warning_threshold: float) -> RiskLevel:
    """แปลงความน่าจะเป็นเป็นระดับความเสี่ยง.

    - P(CRITICAL) >= critical_threshold → CRITICAL  (ตั้งต่ำเพื่อเน้น Recall)
    - P(WARNING) + P(CRITICAL) >= warning_threshold → WARNING
    - นอกนั้น → NORMAL
    """
    p_crit = probs.get(RiskLevel.CRITICAL, 0.0)
    p_warn = probs.get(RiskLevel.WARNING, 0.0)
    if p_crit >= critical_threshold:
        return RiskLevel.CRITICAL
    if p_warn + p_crit >= warning_threshold:
        return RiskLevel.WARNING
    return RiskLevel.NORMAL


class RiskClassifier(ABC):
    name: str = "base"

    def __init__(self, critical_threshold: float = 0.35, warning_threshold: float = 0.5):
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> list[Proba]:
        """คืนค่าความน่าจะเป็นของทั้ง 3 ระดับสำหรับแต่ละข้อความ."""

    def explain(self, text: str) -> list[str]:
        """คำ/วลีที่ทำให้โมเดลตัดสิน (override ได้ เช่น LIME/SHAP/attention)."""
        return []

    def predict_windows(self, windows: Sequence[Window]) -> list[RiskResult]:
        probas = self.predict_proba([w.text for w in windows])
        results = []
        for w, p in zip(windows, probas):
            level = decide_level(p, self.critical_threshold, self.warning_threshold)
            evidence = self.explain(w.text) if level != RiskLevel.NORMAL else []
            results.append(RiskResult(level, p, n_tokens=w.n_tokens, n_turns=w.n_turns, evidence=evidence))
        return results

    def predict(self, texts: Sequence[str]) -> list[RiskResult]:
        windows = [Window(text=t, n_turns=1, n_tokens=count_tokens(t)) for t in texts]
        return self.predict_windows(windows)


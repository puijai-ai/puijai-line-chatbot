"""Baseline 1: Rule-based / Keyword matching (เทียบเท่าวิธีเดิมของปุยใจ).

ใช้เป็นตัวจำลองให้ฝั่งระบบ (คนที่ B) พัฒนาต่อได้ทันที และเป็น baseline ในตารางผลการทดลอง
ข้อจำกัดที่ตั้งใจให้เห็นในเล่ม: ไม่เข้าใจบริบท เช่น "เหนื่อยมาก แต่สนุก" ก็ถูกจับเป็น WARNING
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..preprocessing import normalize_text
from ..schemas import RiskLevel
from .base import Proba, RiskClassifier

CRITICAL_PHRASES = [
    "อยากตาย",
    "ฆ่าตัวตาย",
    "ไม่อยากอยู่แล้ว",
    "ไม่อยากมีชีวิต",
    "จบชีวิต",
    "ทำร้ายตัวเอง",
    "กรีดแขน",
    "หายไปจากโลก",
    "ลาก่อนทุกคน",
    "กินยาเกินขนาด",
    "ไม่อยากตื่นขึ้นมา",
]

WARNING_PHRASES = [
    "เหนื่อย",
    "ท้อ",
    "สิ้นหวัง",
    "ไร้ค่า",
    "ไม่มีใครรัก",
    "ไม่มีใครเข้าใจ",
    "นอนไม่หลับ",
    "เบื่อทุกอย่าง",
    "ร้องไห้",
    "เศร้า",
    "เครียด",
    "โดดเดี่ยว",
    "เป็นภาระ",
    "ไม่มีความสุข",
]

# สำนวนเปรียบเปรยที่ไม่ใช่จุดเสี่ยง เช่น "หิวจะตาย", "ขำจนตาย"
FIGURATIVE_RE = re.compile(r"(หิว|เหนื่อย|ขำ|ร้อน|หนาว|ง่วง|อร่อย|สนุก|เบื่อ|เมื่อย)\s*(จะ|จน)\s*ตาย")

CRITICAL_WEIGHT = 3.0
WARNING_WEIGHT = 1.0
NORMAL_PRIOR = 1.0


class KeywordClassifier(RiskClassifier):
    name = "keyword"

    def _matches(self, text: str) -> tuple[list[str], list[str]]:
        clean = FIGURATIVE_RE.sub(" ", normalize_text(text))
        crit = [p for p in CRITICAL_PHRASES if p in clean]
        warn = [p for p in WARNING_PHRASES if p in clean]
        return crit, warn

    def predict_proba(self, texts: Sequence[str]) -> list[Proba]:
        out = []
        for text in texts:
            crit, warn = self._matches(text)
            scores = {
                RiskLevel.NORMAL: NORMAL_PRIOR,
                RiskLevel.WARNING: WARNING_WEIGHT * len(warn),
                RiskLevel.CRITICAL: CRITICAL_WEIGHT * len(crit),
            }
            total = sum(scores.values())
            out.append({k: v / total for k, v in scores.items()})
        return out

    def explain(self, text: str) -> list[str]:
        crit, warn = self._matches(text)
        return crit + warn


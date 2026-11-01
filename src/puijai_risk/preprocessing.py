"""ทำความสะอาดข้อความภาษาไทยและนับจำนวนคำ.

"คำ" ในโปรเจกต์นี้ = ผลการตัดคำด้วย PyThaiNLP (engine ``newmm``)
ถ้ายังไม่ได้ติดตั้ง PyThaiNLP จะ fallback เป็นการแยกด้วยช่องว่าง (ใช้ได้แค่ตอนทดสอบ)
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_REPEAT_RE = re.compile(r"(.)\1{3,}")  # อักษรซ้ำ >= 4 ตัว เช่น "มากกกกก", "55555"
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize Unicode, ลบ URL, ย่ออักษรซ้ำเหลือ 3 ตัว, รวมช่องว่าง."""
    text = unicodedata.normalize("NFC", text or "")
    text = _URL_RE.sub(" ", text)
    text = _REPEAT_RE.sub(lambda m: m.group(1) * 3, text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


@lru_cache(maxsize=1)
def _thai_tokenizer():
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError:  # pragma: no cover - ขึ้นกับ environment
        return None
    return word_tokenize


def tokenize(text: str) -> list[str]:
    """ตัดคำภาษาไทย (newmm) และตัดช่องว่างทิ้ง."""
    text = normalize_text(text)
    if not text:
        return []
    word_tokenize = _thai_tokenizer()
    if word_tokenize is None:
        return text.split()
    return [t for t in word_tokenize(text, engine="newmm", keep_whitespace=False) if t.strip()]


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def to_wangchanberta_input(text: str, space_token: str = "<_>") -> str:
    """เตรียมข้อความให้ WangchanBERTa ตามแนวทาง ``thai2transformers.process_transformers``:
    ตัวพิมพ์เล็ก + แทนช่องว่างด้วย ``<_>`` (ต้องใช้ฟังก์ชันเดียวกันทั้งตอนเทรนและตอนใช้งาน)
    """
    return normalize_text(text).lower().replace(" ", space_token)


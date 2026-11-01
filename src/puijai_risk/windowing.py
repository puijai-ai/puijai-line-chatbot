"""Sliding Context Window — รวม N ข้อความล่าสุดของผู้ใช้ก่อนส่งเข้าโมเดล.

นี่คือหัวใจของคำถามวิจัย "ต้องใช้กี่ข้อความ / กี่คำ จึงตัดสินจุดเสี่ยงได้แม่นยำที่สุด"
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .preprocessing import tokenize

DEFAULT_SEP = " | "


@dataclass(frozen=True)
class Window:
    text: str
    n_turns: int
    n_tokens: int


def build_window(
    turns: Sequence[str],
    n_turns: int = 3,
    max_tokens: int | None = None,
    sep: str = DEFAULT_SEP,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> Window:
    """รวม ``n_turns`` ข้อความล่าสุด (เรียงเก่า → ใหม่).

    ถ้ากำหนด ``max_tokens`` จะตัดข้อความที่เก่าที่สุดออกจนจำนวนคำไม่เกิน
    (แต่จะเก็บข้อความล่าสุดไว้เสมออย่างน้อย 1 ข้อความ)
    """
    if n_turns < 1:
        raise ValueError("n_turns must be >= 1")
    tok = tokenizer or tokenize
    selected = [t for t in turns[-n_turns:] if t and t.strip()]
    if not selected:
        return Window(text="", n_turns=0, n_tokens=0)

    counts = [len(tok(t)) for t in selected]
    if max_tokens is not None:
        while len(selected) > 1 and sum(counts) > max_tokens:
            selected.pop(0)
            counts.pop(0)
    return Window(text=sep.join(selected), n_turns=len(selected), n_tokens=sum(counts))


def prefix_windows(
    turns: Sequence[str],
    n_turns: int,
    max_tokens: int | None = None,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> list[Window]:
    """จำลองการแชทแบบ real-time: window หลังข้อความที่ 1, 2, ..., k (ใช้คำนวณ ERDE)."""
    return [
        build_window(turns[: k + 1], n_turns=n_turns, max_tokens=max_tokens, tokenizer=tokenizer)
        for k in range(len(turns))
    ]


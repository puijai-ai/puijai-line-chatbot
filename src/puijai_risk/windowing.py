"""Sliding Context Window — รวม K ข้อความล่าสุดของผู้ใช้ก่อนส่งเข้าโมเดล.

รองรับทั้งการนับคำภาษาไทย (Word-level) สำหรับโมเดล Baseline
และการนับโทเคนย่อย SentencePiece (Subword Token-level) ตามสถาปัตยกรรม WangchanBERTa
พร้อม Left-truncation เพื่อรักษาข้อความล่าสุด (Tail preservation)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .preprocessing import normalize_text, tokenize

DEFAULT_SEP = " | "


@dataclass(frozen=True)
class WindowConfig:
    k: int
    max_length: int = 128


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

    ถ้ากำหนด ``max_tokens`` จะตัดข้อความที่เก่าที่สุดออกจากซ้ายจนจำนวนคำไม่เกิน
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


def build_subword_window(
    user_messages: Sequence[str],
    tokenizer: Any,
    config: WindowConfig,
    preprocess: Callable[[str], str] = normalize_text,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """สร้าง Window ข้อมูลนำเข้าระดับ SentencePiece Subword Token สำหรับ Transformer Models.

    - คัดกรอง K ข้อความล่าสุด
    - สำรองพื้นที่สำหรับ Special Tokens (<s>, </s>) ตาม Tokenizer ของโมเดล
    - เชื่อมบริบทระหว่างรอบสนทนาด้วย sep_token_id
    - ตัดข้อมูลเนื้อหาจากด้านซ้าย (Left Truncation) เพื่อคงข้อความรอบล่าสุดไว้เสมอ
    - ส่งคืน Tensor พร้อม Metadata สำหรับตรวจสอบการตัดทอน (Truncation Audit)
    """
    import torch

    if config.k < 1:
        raise ValueError("k must be >= 1")
    special_count = getattr(tokenizer, "num_special_tokens_to_add", lambda pair=False: 2)(pair=False)
    budget = config.max_length - special_count
    if budget < 1:
        raise ValueError(f"max_length ({config.max_length}) is too small for special tokens ({special_count})")

    selected = [m for m in user_messages[-config.k:] if isinstance(m, str) and m.strip()]
    if not selected:
        raise ValueError("No valid user messages found")

    sep_id = getattr(tokenizer, "sep_token_id", None)
    content_ids: list[int] = []

    for index, text in enumerate(selected):
        clean_text = preprocess(text)
        if index > 0 and sep_id is not None:
            content_ids.append(sep_id)
        enc_ids = tokenizer.encode(clean_text, add_special_tokens=False, truncation=False)
        content_ids.extend(enc_ids)

    before_truncation = len(content_ids)
    # Left-truncation: เก็บ budget โทเคนล่าสุด
    kept_ids = content_ids[-budget:]
    
    # เพิ่ม Special Tokens (<s> ... </s>)
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        input_ids = tokenizer.build_inputs_with_special_tokens(kept_ids)
    else:
        bos = getattr(tokenizer, "bos_token_id", 1)
        eos = getattr(tokenizer, "eos_token_id", 2)
        input_ids = [bos] + kept_ids + [eos]

    if len(input_ids) > config.max_length:
        input_ids = input_ids[: config.max_length]

    attention_mask = [1] * len(input_ids)
    metadata = {
        "requested_k": config.k,
        "selected_turns": len(selected),
        "max_length": config.max_length,
        "content_tokens_before": before_truncation,
        "input_tokens_after": len(input_ids),
        "is_truncated": before_truncation > budget,
    }

    encoded = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }
    return encoded, metadata


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

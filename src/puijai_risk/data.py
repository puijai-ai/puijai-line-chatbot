"""โหลดชุดข้อมูลบทสนทนาและสร้างตัวอย่างแบบ Sliding Window.

รูปแบบไฟล์ CSV (UTF-8):
    conversation_id,turn,text,label
    c01,1,สวัสดีปุยใจ,NORMAL

``label`` = ระดับความเสี่ยง ณ ข้อความนั้น (โดยดูบริบทก่อนหน้าประกอบ) ตาม docs/LABELING_GUIDELINE.md
"""

from __future__ import annotations

import csv
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .preprocessing import tokenize
from .schemas import RiskLevel
from .windowing import build_window


@dataclass(frozen=True)
class Turn:
    text: str
    label: RiskLevel


Conversations = dict[str, list[Turn]]


@dataclass(frozen=True)
class WindowSample:
    conversation_id: str
    turn_index: int  # 1-based
    text: str
    label: RiskLevel
    n_turns: int
    n_tokens: int  # จำนวนคำทั้ง window
    current_turn_tokens: int  # จำนวนคำของข้อความล่าสุดอย่างเดียว


def load_conversations(path: str | Path) -> Conversations:
    rows: dict[str, list[tuple[int, Turn]]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            turn = Turn(text=row["text"].strip(), label=RiskLevel.parse(row["label"]))
            rows.setdefault(row["conversation_id"].strip(), []).append((int(row["turn"]), turn))
    return {cid: [t for _, t in sorted(items, key=lambda x: x[0])] for cid, items in rows.items()}


def conversation_gold(turns: list[Turn]) -> RiskLevel:
    """ระดับความเสี่ยงสูงสุดของบทสนทนา (ใช้ใน ERDE)."""
    return max(t.label for t in turns)


def save_conversations(convs: Conversations, path: str | Path) -> None:
    """บันทึกกลับเป็น CSV รูปแบบเดียวกับ load_conversations (ใช้เก็บ test split)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conversation_id", "turn", "text", "label"])
        for cid, turns in convs.items():
            for i, t in enumerate(turns, start=1):
                w.writerow([cid, i, t.text, t.label.name])


def make_window_samples(
    convs: Conversations,
    n_turns: int,
    max_tokens: int | None = None,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> list[WindowSample]:
    tok = tokenizer or tokenize
    samples: list[WindowSample] = []
    for cid, turns in convs.items():
        texts = [t.text for t in turns]
        for k, turn in enumerate(turns):
            w = build_window(texts[: k + 1], n_turns=n_turns, max_tokens=max_tokens, tokenizer=tok)
            samples.append(
                WindowSample(
                    conversation_id=cid,
                    turn_index=k + 1,
                    text=w.text,
                    label=turn.label,
                    n_turns=w.n_turns,
                    n_tokens=w.n_tokens,
                    current_turn_tokens=len(tok(turn.text)),
                )
            )
    return samples


def split_by_conversation(
    convs: Conversations, val_ratio: float = 0.1, test_ratio: float = 0.2, seed: int = 42
) -> tuple[Conversations, Conversations, Conversations]:
    """แบ่ง train/val/test ตาม conversation (กันข้อมูลรั่วระหว่างชุด)."""
    ids = sorted(convs)
    random.Random(seed).shuffle(ids)
    n_test = max(1, round(len(ids) * test_ratio)) if len(ids) > 2 else 0
    n_val = max(1, round(len(ids) * val_ratio)) if len(ids) > 3 else 0
    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]
    pick = lambda keys: {k: convs[k] for k in keys}  # noqa: E731
    return pick(train_ids), pick(val_ids), pick(test_ids)

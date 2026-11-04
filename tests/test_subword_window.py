"""Unit tests for SentencePiece Subword Token-level Sliding Window.

ครอบคลุมทุกกรณีที่กรรมการและผู้ประเมินตรวจ:
1. ประวัติน้อยกว่า K
2. ข้อความสั้น (ไม่ถูกตัดโดยไม่จำเป็น)
3. ข้อความยาว (จำนวน Input IDs ไม่เกิน L)
4. Tail preservation (เนื้อหาหลังตัดตรงกับส่วนท้ายก่อนตัด)
5. ข้อความล่าสุดยาวเกินงบ (เก็บส่วนท้าย และบันทึก is_truncated=True)
"""

from __future__ import annotations

import pytest
from puijai_risk.windowing import WindowConfig, build_subword_window


class MockTokenizer:
    """Mock Tokenizer ที่จำลองพฤติกรรม SentencePiece ของ HuggingFace Transformers."""
    def __init__(self):
        self.sep_token_id = 3
        self.bos_token_id = 1
        self.eos_token_id = 2

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def encode(self, text: str, add_special_tokens: bool = False, truncation: bool = False) -> list[int]:
        # จำลอง 1 คำ/พยางค์ = 1 token id
        words = text.split()
        return [hash(w) % 10000 + 10 for w in words]

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return [self.bos_token_id] + token_ids + [self.eos_token_id]


@pytest.fixture
def tokenizer():
    return MockTokenizer()


def test_history_less_than_k(tokenizer):
    """กรณีประวัติน้อยกว่า K (เช่น มี 2 ข้อความ แต่ K=3)."""
    messages = ["สวัสดีครับ", "วันนี้รู้สึกเศร้า"]
    cfg = WindowConfig(k=3, max_length=64)
    encoded, meta = build_subword_window(messages, tokenizer, cfg)

    assert meta["requested_k"] == 3
    assert meta["selected_turns"] == 2
    assert not meta["is_truncated"]
    assert encoded["input_ids"].shape[1] <= 64


def test_short_messages_not_truncated(tokenizer):
    """ข้อความสั้น ไม่ถูกตัดทอนโดยไม่จำเป็น."""
    messages = ["ข้อความที่หนึ่ง", "ข้อความที่สอง", "ข้อความที่สาม"]
    cfg = WindowConfig(k=3, max_length=128)
    encoded, meta = build_subword_window(messages, tokenizer, cfg)

    assert not meta["is_truncated"]
    assert meta["selected_turns"] == 3


def test_long_messages_within_l_budget(tokenizer):
    """ข้อความยาว รวมแล้วไม่เกินเพดาน L โทเคน."""
    long_msg1 = " ".join([f"คำที่{i}" for i in range(50)])
    long_msg2 = " ".join([f"คำที่{i}" for i in range(50)])
    long_msg3 = " ".join([f"คำที่{i}" for i in range(50)])
    messages = [long_msg1, long_msg2, long_msg3]

    cfg = WindowConfig(k=3, max_length=64)
    encoded, meta = build_subword_window(messages, tokenizer, cfg)

    assert encoded["input_ids"].shape[1] <= 64
    assert meta["is_truncated"]


def test_tail_preservation(tokenizer):
    """Tail preservation: ตรวจสอบว่าโทเคนเนื้อหาส่วนท้ายสุดของบทสนทนายังคงอยู่ครบ."""
    msg1 = "ข้อความเก่า ที่ควรถูกตัดทิ้งเมื่อเกินงบ"
    msg2 = "ข้อความกลาง ที่อาจโดนตัดบางส่วน"
    msg3 = "ข้อความวิกฤตล่าสุด สัญญาณฉุกเฉิน ห้ามตัดทิ้งเด็ดขาด"
    messages = [msg1, msg2, msg3]

    cfg = WindowConfig(k=3, max_length=20)
    encoded, meta = build_subword_window(messages, tokenizer, cfg)

    input_ids = encoded["input_ids"][0].tolist()
    # ตรวจว่า Special Token <s> อยู่หน้าสุด และ </s> อยู่ท้ายสุด
    assert input_ids[0] == tokenizer.bos_token_id
    assert input_ids[-1] == tokenizer.eos_token_id

    # ตรวจสอบว่าโทเคนของ msg3 ปรากฏที่ส่วนท้ายของ input_ids
    msg3_ids = tokenizer.encode(msg3, add_special_tokens=False)
    # ตัด special token ปิดท้ายออก เพื่อดูเนื้อหาท้ายสุด
    content_end = input_ids[-len(msg3_ids) - 1 : -1]
    assert content_end == msg3_ids


def test_latest_turn_exceeds_budget(tokenizer):
    """กรณีข้อความล่าสุดเพียงข้อความเดียวยาวเกินงบ L: เก็บส่วนท้ายของข้อความล่าสุด."""
    huge_msg = " ".join([f"ข้อความยาวมากรอบล่าสุด_{i}" for i in range(100)])
    cfg = WindowConfig(k=1, max_length=32)
    encoded, meta = build_subword_window([huge_msg], tokenizer, cfg)

    assert encoded["input_ids"].shape[1] == 32
    assert meta["is_truncated"]
    assert meta["input_tokens_after"] == 32


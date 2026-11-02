"""การทดสอบหน่วย (Unit Tests) สำหรับการปิด 7 จุดโหว่ของระบบแชทบอทปุยใจ."""

import os
import tempfile
import time
from app.messages import crisis_flex_message, warning_flex_message, build_reply_messages
from app.session_store import InMemorySessionStore
from puijai_risk.safety.policy import SafetyAction
from puijai_risk.schemas import RiskLevel


def test_crisis_flex_button_color_is_red():
    """จุดโหว่ที่ 1: ตรวจสอบว่าปุ่มโทร 1323 เป็นสีแดงสดฉุกเฉิน #EF4444 ตามเกณฑ์เล่มวิจัย."""
    msg = crisis_flex_message()
    footer_buttons = msg["contents"]["footer"]["contents"]
    btn_1323 = footer_buttons[0]
    assert btn_1323["color"] == "#EF4444"
    assert btn_1323["action"]["uri"] == "tel:1323"
    assert "1323" in btn_1323["action"]["label"]


def test_session_store_sqlite_persistence():
    """จุดโหว่ที่ 2: ตรวจสอบว่าเมื่อรีสตาร์ทเซิร์ฟเวอร์ ประวัติบริบท Sliding Window ไม่สูญหาย."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    try:
        now = [1000.0]
        store1 = InMemorySessionStore(max_turns=3, ttl_seconds=300, clock=lambda: now[0], db_path=db_path)
        store1.append("user_abc", "ข้อความที่ 1")
        store1.append("user_abc", "ข้อความที่ 2")
        store1.append("user_abc", "ข้อความที่ 3")

        # จำลองการรีสตาร์ท Server โดยสร้าง Store ใหม่ที่ชี้ไปที่ DB เดิม
        store2 = InMemorySessionStore(max_turns=3, ttl_seconds=300, clock=lambda: now[0], db_path=db_path)
        recovered = store2.get("user_abc")
        assert len(recovered) == 3
        assert recovered == ["ข้อความที่ 1", "ข้อความที่ 2", "ข้อความที่ 3"]
    finally:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except Exception:
            pass


def test_session_store_rate_limiting():
    """จุดโหว่ที่ 7: ตรวจสอบ Rate Limiting ป้องกัน DoS และการสแปมยิงโมเดล."""
    now = [100.0]
    store = InMemorySessionStore(clock=lambda: now[0])
    # ครั้งแรกต้องผ่าน
    assert store.is_rate_limited("user_spam", min_seconds=1.0) is False
    # ยิงซ้ำทันที (เวลาเดิม) ต้องโดนบล็อก
    assert store.is_rate_limited("user_spam", min_seconds=1.0) is True
    # ผ่านไป 1.5 วินาที ต้องผ่านได้ปกติ
    now[0] = 101.5
    assert store.is_rate_limited("user_spam", min_seconds=1.0) is False


def test_session_store_level1_escalation():
    """จุดโหว่ที่ 5: ตรวจสอบการนับสะสม Level 1 เพื่อยกระดับการช่วยเหลือฉุกเฉิน."""
    store = InMemorySessionStore()
    assert store.get_level1_count("user_xyz") == 0
    assert store.increment_level1("user_xyz") == 1
    assert store.increment_level1("user_xyz") == 2
    assert store.increment_level1("user_xyz") == 3
    # รีเซ็ตเมื่ออารมณ์ดีขึ้นหรือพ้นวิกฤต
    store.reset_level1("user_xyz")
    assert store.get_level1_count("user_xyz") == 0


def test_webhook_deduplication():
    """จุดโหว่ที่ 3: ตรวจสอบว่าป้องกันการประมวลผล Reply Token ซ้ำซ้อน."""
    store = InMemorySessionStore()
    token = "line_reply_token_12345"
    assert store.is_token_processed(token) is False
    store.mark_token_processed(token)
    assert store.is_token_processed(token) is True

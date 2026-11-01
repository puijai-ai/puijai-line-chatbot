"""Safety Policy: แปลงระดับความเสี่ยงเป็นการตอบสนองของแชทบอท (คนที่ B)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..schemas import RiskLevel, RiskResult

HOTLINES = [
    {"label": "สายด่วนสุขภาพจิต 1323", "tel": "1323"},
    {"label": "สมาคมสะมาริตันส์ 02-713-6793", "tel": "027136793"},
    {"label": "เหตุฉุกเฉิน 1669", "tel": "1669"},
]

REPLIES = {
    RiskLevel.NORMAL: "ปุยใจรับฟังอยู่นะ เล่าให้ฟังต่อได้เลย",
    RiskLevel.WARNING: (
        "ฟังดูเหมือนช่วงนี้คุณต้องแบกเรื่องหนักๆ อยู่ ขอบคุณที่เล่าให้ปุยใจฟังนะ "
        "ถ้าสะดวก ลองทำแบบประเมินความเครียดสั้นๆ (2Q) ด้วยกันไหม "
        "จะช่วยให้เข้าใจความรู้สึกตัวเองมากขึ้น"
    ),
    RiskLevel.CRITICAL: (
        "ปุยใจเป็นห่วงคุณมากนะ ตอนนี้คุณไม่ได้อยู่คนเดียว "
        "อยากชวนให้คุยกับผู้เชี่ยวชาญที่พร้อมรับฟังตลอด 24 ชั่วโมง กดปุ่มด้านล่างเพื่อโทรได้ทันที "
        "ถ้าตอนนี้อยู่ในอันตราย โทร 1669 ได้เลย"
    ),
}


@dataclass
class SafetyAction:
    level: RiskLevel
    reply_text: str
    show_crisis_card: bool  # ส่ง Flex การ์ดพร้อมปุ่มโทรสายด่วน
    notify_staff: bool  # แจ้งผู้ดูแล/นักจิตวิทยา
    suggest_screening: bool  # ชวนทำแบบคัดกรอง 2Q/9Q
    delegate_to_chatbot: bool  # ให้บทสนทนาหลักของปุยใจตอบต่อตามปกติ

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = self.level.name
        return d


def decide_action(result: RiskResult) -> SafetyAction:
    level = result.level
    return SafetyAction(
        level=level,
        reply_text=REPLIES[level],
        show_crisis_card=level == RiskLevel.CRITICAL,
        notify_staff=level == RiskLevel.CRITICAL,
        suggest_screening=level == RiskLevel.WARNING,
        delegate_to_chatbot=level == RiskLevel.NORMAL,
    )


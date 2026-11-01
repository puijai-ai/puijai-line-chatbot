"""สร้างข้อความตอบกลับ LINE (Text + Flex การ์ดวิกฤต)."""

from __future__ import annotations

from puijai_risk.safety.policy import HOTLINES, SafetyAction


def text_message(text: str) -> dict:
    return {"type": "text", "text": text[:5000]}


def crisis_flex_message() -> dict:
    buttons = [
        {
            "type": "button",
            "style": "primary" if i == 0 else "secondary",
            "height": "sm",
            "color": "#7C9CBF" if i == 0 else None,
            "action": {"type": "uri", "label": h["label"][:40], "uri": f"tel:{h['tel']}"},
        }
        for i, h in enumerate(HOTLINES)
    ]
    for b in buttons:
        if b["color"] is None:
            del b["color"]
    return {
        "type": "flex",
        "altText": "ช่องทางขอความช่วยเหลือ: สายด่วนสุขภาพจิต 1323",
        "contents": {
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "md",
                "contents": [
                    {"type": "text", "text": "คุณไม่ได้อยู่คนเดียวนะ", "weight": "bold", "size": "lg", "wrap": True},
                    {
                        "type": "text",
                        "text": "มีผู้เชี่ยวชาญพร้อมรับฟังคุณตลอด 24 ชั่วโมง ฟรี และเป็นความลับ",
                        "size": "sm",
                        "color": "#555555",
                        "wrap": True,
                    },
                ],
            },
            "footer": {"type": "box", "layout": "vertical", "spacing": "sm", "contents": buttons},
        },
    }


def build_reply_messages(action: SafetyAction) -> list[dict]:
    messages = [text_message(action.reply_text)]
    if action.show_crisis_card:
        messages.append(crisis_flex_message())
    return messages


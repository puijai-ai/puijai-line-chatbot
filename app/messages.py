"""สร้างข้อความตอบกลับ LINE (Text + Flex การ์ดฉุกเฉิน 1323 และการ์ดดูแลใจสีเหลือง).

มาตรฐาน UX/UI Trauma-Informed Design:
- ระดับ 2 (วิกฤต): การ์ดสีแดงสด (#EF4444) พร้อมปุ่มกดโทรออกสายด่วน 1323 (tel:1323) และ 1669
- ระดับ 1 (เสี่ยงเริ่มต้น): การ์ดสีเหลืองอำพัน (#F59E0B) ประคองอารมณ์และชวนทำแบบคัดกรอง 2Q
- ระดับ 0 (ปกติ): ข้อความปลอบโยนทั่วไป
"""

from __future__ import annotations

from puijai_risk.safety.policy import HOTLINES, SafetyAction


def text_message(text: str) -> dict:
    return {"type": "text", "text": text[:5000]}


def crisis_flex_message() -> dict:
    """การ์ดแจ้งเตือนฉุกเฉินสีแดง (#EF4444) สำหรับส่งต่อสายด่วนสุขภาพจิต 1323."""
    buttons = [
        {
            "type": "button",
            "style": "primary",
            "height": "sm",
            "color": "#EF4444",  # สีแดงฉุกเฉินตามมาตรฐานเล่มวิจัย
            "action": {
                "type": "uri",
                "label": "📞 โทรสายด่วน 1323 (โทรฟรี)",
                "uri": "tel:1323",
            },
        },
        {
            "type": "button",
            "style": "primary",
            "height": "sm",
            "color": "#5B9BD5",  # สีฟ้าการแพทย์ฉุกเฉิน 1669
            "action": {
                "type": "uri",
                "label": "🚑 โทรฉุกเฉินการแพทย์ 1669",
                "uri": "tel:1669",
            },
        },
        {
            "type": "button",
            "style": "secondary",
            "height": "sm",
            "action": {
                "type": "uri",
                "label": "สมาคมสะมาริตันส์ 02-713-6793",
                "uri": "tel:027136793",
            },
        },
    ]

    return {
        "type": "flex",
        "altText": "🚨 ช่องทางขอความช่วยเหลือฉุกเฉิน: สายด่วนสุขภาพจิต 1323",
        "contents": {
            "type": "bubble",
            "header": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#EF4444",
                "paddingAll": "lg",
                "contents": [
                    {
                        "type": "text",
                        "text": "🚨 ระบบดูแลใจฉุกเฉิน (Crisis Support)",
                        "weight": "bold",
                        "color": "#FFFFFF",
                        "size": "md",
                    },
                    {
                        "type": "text",
                        "text": "ปุยใจเป็นห่วงความปลอดภัยของคุณมากที่สุดนะครับ",
                        "color": "#FEE2E2",
                        "size": "xs",
                        "margin": "sm",
                    },
                ],
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "md",
                "contents": [
                    {
                        "type": "text",
                        "text": "คุณไม่ได้อยู่คนเดียวนะครับ",
                        "weight": "bold",
                        "size": "md",
                        "color": "#1E293B",
                        "wrap": True,
                    },
                    {
                        "type": "text",
                        "text": "เราอยากให้คุณได้รับความช่วยเหลือจากผู้เชี่ยวชาญที่พร้อมรับฟังตลอด 24 ชั่วโมง ฟรี และเป็นความลับ กดปุ่มด้านล่างเพื่อโทรออกได้ทันที",
                        "size": "sm",
                        "color": "#475569",
                        "wrap": True,
                    },
                    {"type": "separator"},
                    {
                        "type": "text",
                        "text": "*ระบบปุยใจเป็นเครื่องมือคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยทางการแพทย์*",
                        "size": "xxs",
                        "color": "#94A3B8",
                        "wrap": True,
                        "style": "italic",
                    },
                ],
            },
            "footer": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "contents": buttons,
            },
        },
    }


def warning_flex_message() -> dict:
    """การ์ดดูแลใจสีเหลืองอำพัน (#F59E0B) สำหรับผู้มีความเสี่ยงระยะเริ่มต้น (Level 1)."""
    return {
        "type": "flex",
        "altText": "💛 การ์ดดูแลใจเป็นพิเศษจากปุยใจ",
        "contents": {
            "type": "bubble",
            "header": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#F59E0B",
                "paddingAll": "md",
                "contents": [
                    {
                        "type": "text",
                        "text": "💛 การ์ดดูแลใจเป็นพิเศษ",
                        "weight": "bold",
                        "color": "#FFFFFF",
                        "size": "sm",
                    }
                ],
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "contents": [
                    {
                        "type": "text",
                        "text": "ปุยใจรับรู้ถึงความเหนื่อยล้าของคุณนะ",
                        "weight": "bold",
                        "size": "sm",
                        "color": "#1E293B",
                        "wrap": True,
                    },
                    {
                        "type": "text",
                        "text": "ถ้าความรู้สึกนี้มันหนักเกินไป ลองฝึกหายใจ 4-7-8 หรือระบายให้ปุยใจฟังต่อได้เสมอนะครับ",
                        "size": "xs",
                        "color": "#475569",
                        "wrap": True,
                    },
                ],
            },
        },
    }


def build_reply_messages(action: SafetyAction, include_warning_card: bool = False) -> list[dict]:
    """สร้างรายการข้อความตอบกลับตามผลการประเมินความปลอดภัย."""
    messages = [text_message(action.reply_text)]
    if action.show_crisis_card:
        messages.append(crisis_flex_message())
    elif include_warning_card:
        messages.append(warning_flex_message())
    return messages

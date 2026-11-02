"""ตรวจลายเซ็น Webhook และส่งข้อความตอบกลับ/ส่งข้อความฉุกเฉินผ่าน LINE Messaging API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging

import httpx

logger = logging.getLogger("puijai.line")

REPLY_URL = "https://api.line.me/v2/bot/message/reply"
PUSH_URL = "https://api.line.me/v2/bot/message/push"


def compute_signature(body: bytes, channel_secret: str) -> str:
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def verify_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    if not signature or not channel_secret:
        return False
    return hmac.compare_digest(compute_signature(body, channel_secret), signature)


async def reply_message(reply_token: str, messages: list[dict], access_token: str) -> bool:
    """ส่งข้อความตอบกลับผ่าน LINE Reply API คืนค่า True เมื่อส่งสำเร็จ."""
    if not access_token:
        logger.warning("LINE_CHANNEL_ACCESS_TOKEN not set — skip reply (dev mode)")
        return True
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                REPLY_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                json={"replyToken": reply_token, "messages": messages[:5]},
            )
            if resp.status_code >= 400:
                logger.error("LINE reply failed: %s %s", resp.status_code, resp.text[:200])
                return False
            return True
    except Exception as exc:
        logger.error("Error executing LINE reply_message: %s", exc)
        return False


async def push_message(user_id: str, messages: list[dict], access_token: str) -> bool:
    """ส่งข้อความตรงถึงผู้ใช้ผ่าน LINE Push API (Fallback ฉุกเฉินเมื่อ Reply Token หมดอายุ)."""
    if not access_token or not user_id or user_id == "anonymous":
        logger.warning("Cannot push message: invalid user_id or access_token missing")
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                PUSH_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                json={"to": user_id, "messages": messages[:5]},
            )
            if resp.status_code >= 400:
                logger.error("LINE emergency push failed: %s %s", resp.status_code, resp.text[:200])
                return False
            logger.info("Successfully sent emergency push message to user: %s", user_id[:8])
            return True
    except Exception as exc:
        logger.error("Error executing LINE push_message: %s", exc)
        return False

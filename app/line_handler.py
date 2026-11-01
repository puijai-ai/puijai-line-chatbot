"""ตรวจลายเซ็น Webhook และส่งข้อความตอบกลับผ่าน LINE Messaging API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging

import httpx

logger = logging.getLogger("puijai.line")

REPLY_URL = "https://api.line.me/v2/bot/message/reply"


def compute_signature(body: bytes, channel_secret: str) -> str:
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def verify_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    if not signature or not channel_secret:
        return False
    return hmac.compare_digest(compute_signature(body, channel_secret), signature)


async def reply_message(reply_token: str, messages: list[dict], access_token: str) -> None:
    if not access_token:
        logger.warning("LINE_CHANNEL_ACCESS_TOKEN not set — skip reply (dev mode)")
        return
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            REPLY_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            json={"replyToken": reply_token, "messages": messages[:5]},
        )
        if resp.status_code >= 400:
            logger.error("LINE reply failed: %s %s", resp.status_code, resp.text[:200])


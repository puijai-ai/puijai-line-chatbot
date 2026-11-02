"""เก็บข้อความล่าสุดของผู้ใช้แต่ละคน (In-Memory + SQLite Persistent Backing) สำหรับสร้าง Sliding Window.

คุณสมบัติความปลอดภัยและวิศวกรรม:
1. Persistent State: ซิงก์ข้อมูลลง SQLite อัตโนมัติ เซิร์ฟเวอร์รีสตาร์ทแล้วบริบทไม่สูญหาย
2. Privacy & PDPA: จัดเก็บโดยใช้ SHA-256 + salt ไม่เก็บ user_id ตรงๆ
3. Rate Limiter: ป้องกัน DoS และการสแปมยิงโมเดลถี่เกินไป
4. Level 1 Escalation Tracker: ตรวจจับสภาวะผู้ใช้ติดลูปความเสี่ยงระดับ 1 ต่อเนื่อง
5. Webhook Deduplication: ป้องกันการประมวลผลซ้ำจากการส่งเบิ้ลของ LINE Messaging API
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger("puijai.session")


def hash_user_id(user_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()


class InMemorySessionStore:
    def __init__(
        self,
        max_turns: int = 10,
        ttl_seconds: int = 1800,
        clock: Callable[[], float] = time.monotonic,
        db_path: str | Path | None = None,
    ):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._data: dict[str, tuple[float, deque[str]]] = {}
        self._rate_limits: dict[str, float] = {}
        self._level1_counts: dict[str, int] = {}
        self._processed_tokens: dict[str, float] = {}
        self._lock = threading.Lock()

        # Database Setup
        if db_path is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "sessions.db")
        else:
            self.db_path = str(db_path)

        self._init_db()
        self._load_from_db()

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        user_hash TEXT PRIMARY KEY,
                        turns_json TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        level1_count INTEGER DEFAULT 0
                    )
                    """
                )
                conn.commit()
        except Exception as exc:
            logger.warning("Could not initialize SQLite session store: %s", exc)

    def _load_from_db(self) -> None:
        """โหลดสถานะเดิมจาก SQLite กลับเข้าสู่ In-Memory Cache เมื่อเซิร์ฟเวอร์เริ่มทำงาน."""
        try:
            now_epoch = time.time()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT user_hash, turns_json, updated_at, level1_count FROM sessions")
                for u_hash, t_json, updated_at, l1_count in cursor.fetchall():
                    if now_epoch - updated_at <= self.ttl_seconds:
                        turns_list = json.loads(t_json)
                        self._data[u_hash] = (self._clock(), deque(turns_list, maxlen=self.max_turns))
                        self._level1_counts[u_hash] = int(l1_count or 0)
            logger.info("Loaded %d active sessions from SQLite persistent storage", len(self._data))
        except Exception as exc:
            logger.warning("Could not restore sessions from SQLite: %s", exc)

    def _sync_to_db(self, user_hash: str, turns: list[str], level1_count: int = 0) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (user_hash, turns_json, updated_at, level1_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_hash) DO UPDATE SET
                        turns_json = excluded.turns_json,
                        updated_at = excluded.updated_at,
                        level1_count = excluded.level1_count
                    """,
                    (user_hash, json.dumps(turns, ensure_ascii=False), time.time(), level1_count),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("Failed to sync session to SQLite: %s", exc)

    def _expire(self, now: float) -> None:
        stale = [k for k, (ts, _) in self._data.items() if now - ts > self.ttl_seconds]
        for k in stale:
            del self._data[k]
            self._level1_counts.pop(k, None)
            self._rate_limits.pop(k, None)

        stale_tokens = [t for t, ts in self._processed_tokens.items() if now - ts > 60.0]
        for t in stale_tokens:
            del self._processed_tokens[t]

    def append(self, key: str, text: str) -> list[str]:
        with self._lock:
            now = self._clock()
            self._expire(now)
            _, turns = self._data.get(key, (now, deque(maxlen=self.max_turns)))
            turns.append(text)
            self._data[key] = (now, turns)
            result = list(turns)
            self._sync_to_db(key, result, self._level1_counts.get(key, 0))
            return result

    def get(self, key: str) -> list[str]:
        with self._lock:
            self._expire(self._clock())
            entry = self._data.get(key)
            return list(entry[1]) if entry else []

    def clear(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._level1_counts.pop(key, None)
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM sessions WHERE user_hash = ?", (key,))
                    conn.commit()
            except Exception:
                pass

    def is_rate_limited(self, key: str, min_seconds: float = 1.0) -> bool:
        """ตรวจสอบว่าผู้ใช้ส่งข้อความถี่เกินกว่า min_seconds หรือไม่ (True = โดน Rate Limit)."""
        with self._lock:
            now = self._clock()
            last_time = self._rate_limits.get(key, 0.0)
            if now - last_time < min_seconds:
                return True
            self._rate_limits[key] = now
            return False

    def increment_level1(self, key: str) -> int:
        """นับจำนวนครั้งที่ผู้ใช้ได้รับความเสี่ยงระดับ 1 ต่อเนื่องกัน."""
        with self._lock:
            count = self._level1_counts.get(key, 0) + 1
            self._level1_counts[key] = count
            turns = list(self._data[key][1]) if key in self._data else []
            self._sync_to_db(key, turns, count)
            return count

    def reset_level1(self, key: str) -> None:
        """รีเซ็ตตัวนับระดับ 1 เมื่อสถานะเปลี่ยน."""
        with self._lock:
            if key in self._level1_counts and self._level1_counts[key] != 0:
                self._level1_counts[key] = 0
                turns = list(self._data[key][1]) if key in self._data else []
                self._sync_to_db(key, turns, 0)

    def get_level1_count(self, key: str) -> int:
        with self._lock:
            return self._level1_counts.get(key, 0)

    def is_token_processed(self, token: str) -> bool:
        """ตรวจสอบว่า replyToken นี้ถูกประมวลผลไปแล้วหรือไม่."""
        if not token:
            return False
        with self._lock:
            return token in self._processed_tokens

    def mark_token_processed(self, token: str) -> None:
        if token:
            with self._lock:
                self._processed_tokens[token] = self._clock()

"""เก็บข้อความล่าสุดของผู้ใช้แต่ละคน (in-memory) สำหรับสร้าง Sliding Window.

- ไม่เก็บ LINE userId ตรงๆ แต่ใช้ SHA-256 + salt
- ลบอัตโนมัติเมื่อไม่มีการคุยเกิน TTL
- โปรดักชันจริงควรเปลี่ยนเป็น Redis (interface เดิม)
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from collections.abc import Callable


def hash_user_id(user_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()


class InMemorySessionStore:
    def __init__(self, max_turns: int = 10, ttl_seconds: int = 1800, clock: Callable[[], float] = time.monotonic):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._data: dict[str, tuple[float, deque[str]]] = {}
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        stale = [k for k, (ts, _) in self._data.items() if now - ts > self.ttl_seconds]
        for k in stale:
            del self._data[k]

    def append(self, key: str, text: str) -> list[str]:
        with self._lock:
            now = self._clock()
            self._expire(now)
            _, turns = self._data.get(key, (now, deque(maxlen=self.max_turns)))
            turns.append(text)
            self._data[key] = (now, turns)
            return list(turns)

    def get(self, key: str) -> list[str]:
        with self._lock:
            self._expire(self._clock())
            entry = self._data.get(key)
            return list(entry[1]) if entry else []

    def clear(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


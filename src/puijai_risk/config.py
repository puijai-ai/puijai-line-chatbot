"""ค่าตั้งของระบบ อ่านจาก environment / ไฟล์ .env"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Risk model
    risk_model: str = "keyword"  # keyword | tfidf | wangchanberta
    model_path: str = "models/wangchanberta-risk"
    n_turns: int = 3
    max_tokens: int = 256
    critical_threshold: float = 0.35
    warning_threshold: float = 0.5

    # Session
    session_max_turns: int = 10
    session_ttl_seconds: int = 1800
    user_id_salt: str = "change-me"

    # LINE
    line_channel_secret: str = ""
    line_channel_access_token: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()


"""สัญญากลาง (interface) ระหว่างฝั่ง AI/ML (คนที่ A) และฝั่งระบบ (คนที่ B)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class RiskLevel(IntEnum):
    NORMAL = 0  # สนทนาทั่วไป / ระบายความเครียดปกติ
    WARNING = 1  # สัญญาณซึมเศร้า ท้อแท้ สิ้นหวัง
    CRITICAL = 2  # มีความคิดทำร้ายตัวเอง/ฆ่าตัวตาย → เข้า safety flow ทันที

    @classmethod
    def parse(cls, value: str | int) -> "RiskLevel":
        """รับได้ทั้ง 0/1/2 และ 'NORMAL'/'WARNING'/'CRITICAL'."""
        if isinstance(value, int):
            return cls(value)
        text = str(value).strip()
        if text.isdigit():
            return cls(int(text))
        return cls[text.upper()]


LABELS: list[RiskLevel] = list(RiskLevel)


@dataclass
class RiskResult:
    level: RiskLevel
    probabilities: dict[RiskLevel, float]
    n_tokens: int = 0  # จำนวนคำที่ใช้ตัดสิน (ใช้ในการทดลองความยาว)
    n_turns: int = 1  # จำนวนข้อความที่รวมใน window
    evidence: list[str] = field(default_factory=list)  # คำ/วลีที่ส่งผล

    def to_dict(self) -> dict:
        return {
            "level": int(self.level),
            "label": self.level.name,
            "probabilities": {k.name: round(v, 4) for k, v in self.probabilities.items()},
            "n_tokens": self.n_tokens,
            "n_turns": self.n_turns,
            "evidence": self.evidence,
        }


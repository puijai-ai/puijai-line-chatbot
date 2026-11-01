"""โมเดลหลัก: WangchanBERTa ที่ fine-tune แล้ว (ต้องติดตั้ง requirements-ml.txt).

โมเดลได้มาจาก ``puijai_risk.training.train_wangchanberta`` (บันทึกเป็นโฟลเดอร์)
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from ..preprocessing import to_wangchanberta_input, tokenize
from ..schemas import LABELS, RiskLevel
from .base import Proba, RiskClassifier

if Path("D:/puijai_env/hf_cache").exists():
    os.environ.setdefault("HF_HOME", "D:/puijai_env/hf_cache")

BASE_MODEL = "airesearch/wangchanberta-base-att-spm-uncased"
MAX_LENGTH = 416  # ความยาวสูงสุดที่ WangchanBERTa รองรับ


class WangchanBertaClassifier(RiskClassifier):
    name = "wangchanberta"

    def __init__(self, model_path: str, device: str | None = None, batch_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        self.batch_size = batch_size

    def predict_proba(self, texts: Sequence[str]) -> list[Proba]:
        torch = self._torch
        out: list[Proba] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [to_wangchanberta_input(t) for t in texts[i : i + self.batch_size]]
            enc = self.tokenizer(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
            out.extend({RiskLevel(j): float(row[j]) for j in range(len(LABELS))} for row in probs)
        return out

    def explain(self, text: str) -> list[str]:
        """Explainable AI (Leave-One-Token-Out Perturbation):
        วัดว่าเมื่อตัดคำแต่ละคำในประโยคออก ความน่าจะเป็นของจุดเสี่ยง (WARNING + CRITICAL) ลดลงเท่าไร
        คำที่ทำให้คะแนนความเสี่ยงลดลงมากที่สุดคือคำสำคัญที่เป็นเหตุผลในการตัดสินของโมเดล
        """
        words = [w for w in tokenize(text) if w != "|" and len(w.strip()) > 1]
        if not words:
            return []
        unique_words = list(dict.fromkeys(words))[:16]
        base_prob = self.predict_proba([text])[0]
        base_risk = base_prob[RiskLevel.CRITICAL] * 2.0 + base_prob[RiskLevel.WARNING]
        perturbed_texts = [text.replace(w, " ", 1) for w in unique_words]
        perturbed_probs = self.predict_proba(perturbed_texts)
        impacts: list[tuple[float, str]] = []
        for w, p in zip(unique_words, perturbed_probs):
            risk = p[RiskLevel.CRITICAL] * 2.0 + p[RiskLevel.WARNING]
            drop = base_risk - risk
            if drop > 0.005:
                impacts.append((drop, w))
        impacts.sort(reverse=True)
        return [w for _, w in impacts[:5]]


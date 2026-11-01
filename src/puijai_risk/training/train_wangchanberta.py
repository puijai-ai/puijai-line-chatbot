"""Fine-tune WangchanBERTa สำหรับจำแนกระดับความเสี่ยง 3 คลาส (คนที่ A).

แนะนำให้รันบน Google Colab (GPU) — ดู notebooks/01_train_colab.ipynb

    python -m puijai_risk.training.train_wangchanberta \
        --data data/processed/conversations.csv --output-dir models/wangchanberta-risk --n-turns 3

- แบ่ง train/val/test ตาม conversation (กันข้อมูลรั่ว) และบันทึกไว้ที่ <output-dir>/splits/
- Weighted Cross-Entropy เพราะข้อมูลจุดเสี่ยงมีน้อย (class imbalance)
- เลือก checkpoint ด้วย score = (Recall ของ CRITICAL + Macro-F1) / 2
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

if Path("D:/puijai_env/hf_cache").exists():
    os.environ.setdefault("HF_HOME", "D:/puijai_env/hf_cache")
    os.environ.setdefault("TEMP", "D:/puijai_env/tmp")
    os.environ.setdefault("TMP", "D:/puijai_env/tmp")

from ..classifiers.base import decide_level
from ..data import load_conversations, make_window_samples, save_conversations, split_by_conversation
from ..evaluation.metrics import classification_report
from ..preprocessing import to_wangchanberta_input
from ..schemas import LABELS, RiskLevel

BASE_MODEL = "airesearch/wangchanberta-base-att-spm-uncased"
MAX_LENGTH = 416


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", default="models/wangchanberta-risk")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--n-turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--critical-threshold", type=float, default=0.35)
    ap.add_argument("--warning-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    convs = load_conversations(args.data)
    train_c, val_c, test_c = split_by_conversation(convs, seed=args.seed)
    out = Path(args.output_dir)
    for name, c in (("train", train_c), ("val", val_c), ("test", test_c)):
        save_conversations(c, out / "splits" / f"{name}.csv")  # ใช้ test.csv กับ length_experiment
    splits = {
        name: make_window_samples(c, args.n_turns, args.max_tokens)
        for name, c in (("train", train_c), ("val", val_c), ("test", test_c))
    }
    print({k: len(v) for k, v in splits.items()})

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label={int(l): l.name for l in LABELS},
        label2id={l.name: int(l) for l in LABELS},
    )

    class RiskDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            texts = [to_wangchanberta_input(s.text) for s in samples]
            self.enc = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
            self.labels = [int(s.label) for s in samples]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    datasets = {k: RiskDataset(v) for k, v in splits.items()}

    # class weights = inverse frequency
    counts = Counter(int(s.label) for s in splits["train"])
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(LABELS) * counts[i]) if counts.get(i) else 1.0 for i in range(len(LABELS))],
        dtype=torch.float,
    )
    print("class weights:", weights.tolist())

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(outputs.logits.device))
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = [
            int(decide_level({RiskLevel(j): float(row[j]) for j in range(len(LABELS))},
                             args.critical_threshold, args.warning_threshold))
            for row in probs
        ]
        rep = classification_report(np.asarray(labels).tolist(), preds)
        return {
            "macro_f1": rep["macro_f1"],
            "critical_recall": rep["critical_recall"],
            "accuracy": rep["accuracy"],
            "selection_score": (rep["critical_recall"] + rep["macro_f1"]) / 2,
        }

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="selection_score",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    from transformers import DataCollatorWithPadding

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"] if len(datasets["val"]) else datasets["test"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    test_metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    (out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    print("test:", test_metrics)


if __name__ == "__main__":
    main()

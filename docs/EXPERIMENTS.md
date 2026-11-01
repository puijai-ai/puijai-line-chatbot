# การออกแบบการทดลอง (บทที่ 4)

## คำถามวิจัย
1. โมเดล WangchanBERTa ตรวจจับจุดเสี่ยงได้ดีกว่าวิธีเดิม (Keyword) และ TF-IDF หรือไม่
2. ข้อความต้องยาว**กี่คำ**โมเดลจึงตัดสินได้แม่นยำ
3. ต้องใช้บริบทย้อนหลัง**กี่ข้อความ (N)** จึงจับจุดเสี่ยงได้ทั้ง "แม่น" และ "เร็ว"

## ตัววัดผล
| ตัววัด | เหตุผล |
|---|---|
| **Recall (CRITICAL)** — ตัวหลัก | ห้ามหลุดคนที่อยู่ในจุดเสี่ยงจริง (ลด False Negative) |
| Precision / F1 (CRITICAL) | ไม่แจ้งเตือนมั่วกับประโยคเปรียบเปรย |
| Macro-F1 | ข้อมูลไม่สมดุล — Accuracy หลอกได้ |
| **ERDE₅ / ERDE₅₀** (ต่ำ = ดี) | วัดทั้งความถูกต้องและความเร็ว (ใช้กี่ข้อความก่อนแจ้งเตือน) — Losada & Crestani (2016) |
| Latency (ms/ข้อความ) | ต้องตอบใน LINE แบบ real-time |

## Exp.0 — เปรียบเทียบโมเดล (N = 3)
| โมเดล | Macro-F1 | Precision (C) | Recall (C) | F1 (C) | ERDE₅ | Latency |
|---|---|---|---|---|---|---|
| Keyword (วิธีเดิม) | | | | | | |
| TF-IDF + LR | | | | | | |
| **WangchanBERTa** | | | | | | |

## Exp.1 — ความยาวข้อความ (N = 1)
| ความยาว (คำ) | n | Macro-F1 | Recall (C) |
|---|---|---|---|
| < 5 | | | |
| 5–20 | | | |
| > 20 | | | |

## Exp.2 — จำนวนข้อความสะสม N
| N | คำเฉลี่ย/window | Macro-F1 | Recall (C) | ERDE₅ | ERDE₅₀ |
|---|---|---|---|---|---|
| 1 | | | | | |
| 3 | | | | | |
| 5 | | | | | |

## Exp.3 (ทางเลือก) — ปรับ threshold
ปรับ `CRITICAL_THRESHOLD` = 0.2 … 0.6 แล้ววาดกราฟ Precision–Recall เพื่อเลือกจุดที่ Recall ≥ 0.9

## วิธีรัน
ใช้ test split ชุดเดียวกันทุกโมเดล (สคริปต์เทรนบันทึกไว้ที่ `<output-dir>/splits/test.csv`)
```bash
TEST=models/wangchanberta-risk/splits/test.csv
python -m puijai_risk.evaluation.length_experiment --classifier keyword --data $TEST
python -m puijai_risk.evaluation.length_experiment --classifier tfidf --model-path models/tfidf.joblib --data $TEST
python -m puijai_risk.evaluation.length_experiment --classifier wangchanberta --model-path models/wangchanberta-risk --data $TEST
```
ผลลัพธ์อยู่ใน `results/length_experiment_<model>_exp1.csv`, `_exp2.csv` และกราฟ `_exp2.png`

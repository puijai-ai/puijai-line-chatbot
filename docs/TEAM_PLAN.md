# แผนการทำงานทีม (2 คน)

| บทบาท | ชื่อ | ความรับผิดชอบ |
|---|---|---|
| **คนที่ A — AI/ML Engineer** | _(ใส่ชื่อ)_ | ข้อมูล, preprocessing, baseline, fine-tune WangchanBERTa, การทดลองความยาวคำ/Sliding Window/ERDE |
| **คนที่ B — System & Integration Engineer** | _(ใส่ชื่อ)_ | FastAPI, LINE webhook, session, safety flow, Flex message, deploy, load test, UAT, แผนภาพ/Figma |

## สัญญาระหว่าง A และ B
- คุยกันผ่าน `RiskClassifier.predict_windows(windows) -> list[RiskResult]` (`src/puijai_risk/classifiers/base.py`)
- B พัฒนาโดยใช้ `KeywordClassifier` ไปก่อน → A ส่งโมเดลเป็นโฟลเดอร์ → สลับด้วย `.env` (`RISK_MODEL=wangchanberta`)
- ถ้าจะเปลี่ยน `RiskResult` / `RiskLevel` ต้องตกลงกันทั้งสองคน + แก้ test

## Milestones (16 สัปดาห์)
| สัปดาห์ | ร่วมกัน | คนที่ A | คนที่ B |
|---|---|---|---|
| 1–2 | ทบทวนวรรณกรรม, คู่มือติดป้าย, ยื่นหัวข้อ CEAI-P1-01 | ศึกษา WangchanBERTa / PyThaiNLP | ศึกษา LINE Messaging API / FastAPI |
| 3–4 | ติดป้ายข้อมูลข้ามกัน + Cohen's Kappa ≥ 0.6 | Preprocessing + Keyword/TF-IDF baseline | Webhook + Session store + `/predict` |
| 5–7 | | Fine-tune WangchanBERTa (Colab) | Safety flow, Flex การ์ด, แผนภาพบน Figma |
| 8–10 | สอบความก้าวหน้า | Exp.1 ความยาวคำ, Exp.2 N ข้อความ + ERDE, ปรับ threshold | เชื่อมโมเดลจริง, Deploy (Cloud Run/Render), วัด latency |
| 11–12 | | Explainability (LIME/SHAP), error analysis | Load test, แจ้งเตือนผู้ดูแล, logging ที่ไม่เก็บข้อความดิบ |
| 13–14 | UAT กับผู้ใช้ทดลอง | สรุปตาราง/กราฟบทที่ 4 | สรุปผลทดสอบระบบบทที่ 4 |
| 15–16 | เขียนเล่ม, สไลด์, ซ้อมสอบ | | |

## Definition of Done (ทุกงาน)
- [ ] `pytest -q` ผ่าน
- [ ] มี test ใหม่สำหรับโค้ดใหม่
- [ ] ไม่มีข้อความดิบของผู้ใช้ใน log / git
- [ ] อัปเดต README หรือ docs ถ้าเปลี่ยนวิธีใช้งาน

## Git workflow
- `main` = เวอร์ชันที่รันได้เสมอ; ทำงานบน branch `a/<feature>` และ `b/<feature>`
- เปิด Pull Request แล้วให้อีกคนรีวิวก่อน merge


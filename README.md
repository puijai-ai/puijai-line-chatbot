# PuijaiRiskNLP — ระบบตรวจจับจุดเสี่ยงภาวะวิกฤตแชทบอทปุยใจ

> **การพัฒนาระบบตรวจจับจุดเสี่ยงภาวะวิกฤตจากบริบทข้อความสนทนาภาษาไทยด้วยโมเดล WangchanBERTa สำหรับแชทบอทปุยใจบนแพลตฟอร์ม LINE**  
> *(Development of a Thai Contextual Crisis Risk Detection System Using WangchanBERTa for the Puijai LINE Chatbot)*  
> **รายวิชา:** 1321205 โครงงานวิศวกรรมคอมพิวเตอร์และปัญญาประดิษฐ์ 1 (PIM)  
> **ภาคการศึกษา:** 1 | **ปีการศึกษา:** 2569 | **กลุ่มเรียน:** 1  

### 👥 คณะผู้จัดทำ (Project Members)
1. **นายศุภโชค ชมภูพื้น** (รหัสนักศึกษา: **6752300852**) — AI & Backend Developer
2. **นายปภังกร มณีรอด** (รหัสนักศึกษา: **6752301158**) — Business Analyst, UX/UI & QA Engineer

### 🧑‍🏫 อาจารย์ที่ปรึกษาโครงงาน (Project Advisor)
* **อาจารย์ สาธิดา สุขพงษ์**

---

## ภาพรวม
```
ผู้ใช้ LINE → /callback → Session Store (N ข้อความล่าสุด) → Sliding Window
            → RiskClassifier (keyword | tfidf | wangchanberta) → Safety Policy
            → NORMAL: ปุยใจคุยต่อตามปกติ | WARNING: ประคับประคอง + 2Q | CRITICAL: การ์ดโทร 1323
```
ระบบปุยใจเดิมเรียกใช้ผ่าน `POST /predict` ได้ด้วย (ไม่ต้องย้ายมาใช้ LINE webhook ตัวนี้)

## เริ่มต้นใช้งาน (Windows / PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy .env.example .env

pytest -q                                   # รันชุดทดสอบ
uvicorn app.main:app --reload               # เปิด http://127.0.0.1:8000/docs
python -m puijai_risk.evaluation.length_experiment --classifier keyword --data data/sample/sample_conversations.csv
```

ทดลองยิง API:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes('{"text":"ช่วงนี้ท้อมาก","user_id":"demo"}'))
```

## เทรนโมเดล (คนที่ A)
ใช้ `requirements-ml.txt` บน **Google Colab (GPU)** หรือ Python 3.11/3.12 — ดู [`notebooks/01_train_colab.ipynb`](notebooks/01_train_colab.ipynb)
```bash
pip install -r requirements-ml.txt && pip install -e .
python -m puijai_risk.training.train_wangchanberta --data data/processed/conversations.csv --output-dir models/wangchanberta-risk --n-turns 3
# สคริปต์เทรนบันทึก train/val/test split ไว้ที่ models/wangchanberta-risk/splits/
python -m puijai_risk.classifiers.tfidf_svm --data models/wangchanberta-risk/splits/train.csv --out models/tfidf.joblib
python -m puijai_risk.evaluation.length_experiment --classifier wangchanberta --model-path models/wangchanberta-risk --data models/wangchanberta-risk/splits/test.csv
```
สลับโมเดลที่ใช้ในแอป: แก้ `.env` → `RISK_MODEL=wangchanberta` และ `MODEL_PATH=models/wangchanberta-risk`

## เชื่อม LINE (คนที่ B)
1. สร้าง Messaging API channel ใน [LINE Developers](https://developers.line.biz/) → ใส่ `LINE_CHANNEL_SECRET` / `LINE_CHANNEL_ACCESS_TOKEN` ใน `.env`
2. `uvicorn app.main:app --port 8000` + `ngrok http 8000`
3. ตั้ง Webhook URL = `https://<ngrok-domain>/callback` แล้วเปิด *Use webhook*

## โครงสร้าง
| Path | เจ้าของ | หน้าที่ |
|---|---|---|
| `src/puijai_risk/schemas.py`, `classifiers/base.py` | A + B | สัญญากลาง `RiskClassifier` / `RiskResult` |
| `src/puijai_risk/preprocessing.py`, `windowing.py`, `data.py` | A | ทำความสะอาด ตัดคำ Sliding Window ชุดข้อมูล |
| `src/puijai_risk/classifiers/` | A | keyword / TF-IDF / WangchanBERTa |
| `src/puijai_risk/training/` | A | Fine-tune WangchanBERTa (weighted loss) |
| `src/puijai_risk/evaluation/` | A | Recall/F1/ERDE + การทดลองความยาวคำ/จำนวนข้อความ |
| `src/puijai_risk/safety/policy.py` | B | ระดับความเสี่ยง → การตอบสนอง |
| `app/` | B | FastAPI, LINE webhook, session, Flex message |
| `docs/` | ร่วม | แบ่งงาน, คู่มือติดป้าย, การทดลอง, แผนภาพ |

## ความปลอดภัยและจริยธรรม
- ไม่ log ข้อความดิบ, hash LINE userId ด้วย salt, `data/raw` และ `.env` ไม่ถูก commit
- ต้องผ่านการพิจารณาจริยธรรมการวิจัยในมนุษย์ก่อนเก็บข้อมูลจริง
- ระบบนี้เป็นเครื่องมือ**คัดกรองเบื้องต้น** ไม่ใช่การวินิจฉัย — กรณีวิกฤตต้องส่งต่อผู้เชี่ยวชาญ (1323)
- `data/sample/` เป็นข้อความสังเคราะห์สำหรับทดสอบ pipeline เท่านั้น

## เอกสารอ้างอิงหลัก
1. Lowphansirikul et al. (2021). *WangchanBERTa: Pretraining transformer-based Thai Language Models.* arXiv:2101.09635
2. Losada & Crestani (2016). *A Test Collection for Research on Depression and Language Use.* CLEF 2016 (นิยาม ERDE)
3. Losada, Crestani & Parapar (2017). *eRISK 2017: CLEF Lab on Early Risk Prediction on the Internet.*
4. Boonyarat et al. *Leveraging Enhanced BERT Models for Detecting Suicidal Ideation in Thai Social Media Content Amidst COVID-19* (SIED-Thai)
5. DMIND (จุฬาฯ + กรมสุขภาพจิต) — *Accuracy and concordance of DMIND AI application…*, Chulalongkorn Medical Journal (2024)
6. กรมสุขภาพจิต — แบบคัดกรอง 2Q / 9Q / 8Q

> [!NOTE]
> ก่อนใส่ในเล่ม ให้ตรวจชื่อผู้แต่ง ปี และแหล่งตีพิมพ์ของข้อ 4–5 กับต้นฉบับอีกครั้ง

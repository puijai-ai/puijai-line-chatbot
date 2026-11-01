import json

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.line_handler import compute_signature
from puijai_risk.config import Settings

SECRET = "test-secret"


@pytest.fixture
def client():
    settings = Settings(_env_file=None, line_channel_secret=SECRET, n_turns=3)
    with TestClient(main_module.create_app(settings=settings)) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "classifier": "keyword"}
    assert client.get("/").status_code == 200
    assert "PuijaiRiskNLP" in client.get("/").text
    assert client.get("/models").json()["available"]["keyword"] is True
    assert "summary" in client.get("/experiments").json()


def test_predict_single_text(client):
    data = client.post("/predict", json={"text": "อยากตาย"}).json()
    assert data["risk"]["label"] == "CRITICAL"
    assert data["action"]["show_crisis_card"] is True


def test_predict_uses_session_context(client):
    client.post("/predict", json={"text": "ช่วงนี้ท้อมาก", "user_id": "U1"})
    data = client.post("/predict", json={"text": "ไม่รู้สิ", "user_id": "U1"}).json()
    # ข้อความล่าสุดไม่มีคำเสี่ยง แต่บริบทก่อนหน้าทำให้ยังเป็น WARNING
    assert data["risk"]["label"] == "WARNING"
    assert data["risk"]["n_turns"] == 2
    assert isinstance(data["tokens"], list) and len(data["tokens"]) > 0
    assert client.post("/session/clear", json={"user_id": "U1"}).json()["status"] == "cleared"


def test_predict_with_history(client):
    data = client.post("/predict", json={"text": "โอเค", "history": ["วันนี้สนุกมาก"]}).json()
    assert data["risk"]["label"] == "NORMAL"


def test_callback_rejects_bad_signature(client):
    r = client.post("/callback", content=b"{}", headers={"X-Line-Signature": "bad"})
    assert r.status_code == 400


def test_callback_replies_with_crisis_card(client, monkeypatch):
    sent = []

    async def fake_reply(token, messages, access_token):
        sent.append((token, messages))

    monkeypatch.setattr(main_module, "reply_message", fake_reply)
    body = json.dumps(
        {
            "events": [
                {
                    "type": "message",
                    "replyToken": "r1",
                    "source": {"userId": "U999"},
                    "message": {"type": "text", "text": "ไม่อยากอยู่แล้ว"},
                }
            ]
        },
        ensure_ascii=False,
    ).encode("utf-8")
    r = client.post("/callback", content=body, headers={"X-Line-Signature": compute_signature(body, SECRET)})
    assert r.status_code == 200
    assert sent[0][0] == "r1"
    assert [m["type"] for m in sent[0][1]] == ["text", "flex"]


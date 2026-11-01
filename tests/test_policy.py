from app.messages import build_reply_messages
from app.session_store import InMemorySessionStore, hash_user_id
from puijai_risk.safety.policy import decide_action
from puijai_risk.schemas import RiskLevel, RiskResult


def result(level: RiskLevel) -> RiskResult:
    return RiskResult(level=level, probabilities={lvl: 1.0 if lvl == level else 0.0 for lvl in RiskLevel})


def test_critical_triggers_crisis_flow():
    a = decide_action(result(RiskLevel.CRITICAL))
    assert a.show_crisis_card and a.notify_staff and not a.delegate_to_chatbot
    msgs = build_reply_messages(a)
    assert msgs[1]["type"] == "flex"
    uris = [b["action"]["uri"] for b in msgs[1]["contents"]["footer"]["contents"]]
    assert "tel:1323" in uris


def test_warning_suggests_screening():
    a = decide_action(result(RiskLevel.WARNING))
    assert a.suggest_screening and not a.show_crisis_card
    assert len(build_reply_messages(a)) == 1


def test_normal_delegates_to_chatbot():
    a = decide_action(result(RiskLevel.NORMAL))
    assert a.delegate_to_chatbot and not a.notify_staff


def test_session_store_window_and_ttl():
    now = [0.0]
    store = InMemorySessionStore(max_turns=2, ttl_seconds=10, clock=lambda: now[0])
    store.append("u", "a")
    store.append("u", "b")
    assert store.append("u", "c") == ["b", "c"]
    now[0] = 100.0
    assert store.get("u") == []


def test_user_id_is_hashed():
    h = hash_user_id("U123", "salt")
    assert h != "U123" and len(h) == 64
    assert h != hash_user_id("U123", "other-salt")


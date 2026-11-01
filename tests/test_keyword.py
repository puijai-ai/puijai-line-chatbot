from puijai_risk.classifiers import KeywordClassifier, decide_level, get_classifier
from puijai_risk.schemas import RiskLevel


def level(text: str) -> RiskLevel:
    return KeywordClassifier().predict([text])[0].level


def test_critical_phrase():
    assert level("ไม่อยากอยู่แล้ว อยากหายไปจากโลกนี้") == RiskLevel.CRITICAL


def test_warning_phrase():
    assert level("ช่วงนี้ท้อ ทำอะไรก็ไม่ดี") == RiskLevel.WARNING


def test_figurative_is_not_risk():
    assert level("หิวจะตายแล้ว") == RiskLevel.NORMAL
    assert level("ขำจนตาย") == RiskLevel.NORMAL


def test_normal():
    assert level("วันนี้อากาศดีจัง") == RiskLevel.NORMAL


def test_evidence_and_probabilities():
    r = KeywordClassifier().predict(["อยากตาย"])[0]
    assert "อยากตาย" in r.evidence
    assert abs(sum(r.probabilities.values()) - 1.0) < 1e-9


def test_decide_level_thresholds():
    p = {RiskLevel.NORMAL: 0.6, RiskLevel.WARNING: 0.0, RiskLevel.CRITICAL: 0.4}
    assert decide_level(p, critical_threshold=0.35, warning_threshold=0.5) == RiskLevel.CRITICAL
    assert decide_level(p, critical_threshold=0.5, warning_threshold=0.5) == RiskLevel.NORMAL
    assert decide_level(p, critical_threshold=0.5, warning_threshold=0.3) == RiskLevel.WARNING


def test_factory():
    assert get_classifier("keyword").name == "keyword"


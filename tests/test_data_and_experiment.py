from pathlib import Path

from puijai_risk.data import load_conversations, make_window_samples, split_by_conversation
from puijai_risk.evaluation.length_experiment import exp1_length_buckets, exp2_context_window
from puijai_risk.classifiers import KeywordClassifier
from puijai_risk.schemas import RiskLevel

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_conversations.csv"


def test_load_and_window_samples():
    convs = load_conversations(SAMPLE)
    assert len(convs) == 12
    samples = make_window_samples(convs, n_turns=3, tokenizer=str.split)
    assert len(samples) == 36
    third = [s for s in samples if s.conversation_id == "c03" and s.turn_index == 3][0]
    assert third.n_turns == 3 and third.label == RiskLevel.CRITICAL


def test_split_has_no_leakage():
    convs = load_conversations(SAMPLE)
    train, val, test = split_by_conversation(convs, seed=1)
    assert set(train).isdisjoint(test) and set(train).isdisjoint(val) and set(val).isdisjoint(test)
    assert len(train) + len(val) + len(test) == len(convs)


def test_experiments_run_end_to_end():
    convs = load_conversations(SAMPLE)
    clf = KeywordClassifier()
    rows1 = exp1_length_buckets(clf, convs, max_tokens=None)
    rows2 = exp2_context_window(clf, convs, [1, 3], None, RiskLevel.CRITICAL)
    assert {r["bucket"] for r in rows1} == {"<5", "5-20", ">20"}
    assert [r["n_turns"] for r in rows2] == [1, 3]
    assert all(0.0 <= r["erde_5"] <= 1.0 for r in rows2)


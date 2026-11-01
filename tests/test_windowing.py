from puijai_risk.windowing import build_window, prefix_windows

ws = str.split  # tokenizer ง่ายๆ สำหรับทดสอบ (ไม่ขึ้นกับ PyThaiNLP)


def test_takes_last_n_turns():
    w = build_window(["a", "b c", "d"], n_turns=2, tokenizer=ws)
    assert w.text == "b c | d"
    assert w.n_turns == 2
    assert w.n_tokens == 3


def test_max_tokens_drops_oldest_but_keeps_latest():
    w = build_window(["a b c d", "e f", "g"], n_turns=3, max_tokens=3, tokenizer=ws)
    assert w.text == "e f | g"
    w = build_window(["a b c d e"], n_turns=3, max_tokens=2, tokenizer=ws)
    assert w.n_turns == 1  # ข้อความล่าสุดต้องอยู่เสมอ


def test_empty_and_invalid():
    assert build_window([], n_turns=3, tokenizer=ws).n_turns == 0
    try:
        build_window(["a"], n_turns=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_prefix_windows_simulates_streaming():
    windows = prefix_windows(["a", "b", "c"], n_turns=2, tokenizer=ws)
    assert [w.text for w in windows] == ["a", "a | b", "b | c"]


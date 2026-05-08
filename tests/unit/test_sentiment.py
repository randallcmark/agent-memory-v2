from agent_memory_v2.sentiment import detect_sentiment


def test_detect_negative_sentiment():
    result = detect_sentiment("I am really frustrated with this bug.")
    assert result.label == "negative"
    assert "negative" in result.cues


def test_detect_distress_sentiment():
    result = detect_sentiment("I feel overwhelmed and anxious about this.")
    assert result.label == "distressed"
    assert "distress" in result.cues


def test_detect_positive_sentiment():
    result = detect_sentiment("Great, thanks for the help.")
    assert result.label == "positive"
    assert "positive" in result.cues


# ── negation suppression ──────────────────────────────────────────────────────


def test_negation_suppresses_negative():
    result = detect_sentiment("I'm not angry at all.")
    assert result.label != "negative"


def test_negation_suppresses_distress():
    result = detect_sentiment("I'm not overwhelmed, I'm fine.")
    assert result.label != "distressed"


def test_negation_not_confused_by_distant_not():
    # "not" is >3 tokens away from "frustrated" — should still trigger
    result = detect_sentiment("That is not what I want but I feel very frustrated about it.")
    assert result.label == "negative"


# ── expanded keyword coverage ─────────────────────────────────────────────────


def test_new_negative_keyword_sad():
    result = detect_sentiment("I feel really sad today.")
    assert result.label == "negative"


def test_new_negative_keyword_stuck():
    result = detect_sentiment("I'm stuck and can't move forward.")
    assert result.label == "negative"


def test_new_distress_keyword_struggling():
    result = detect_sentiment("I'm struggling to keep up with everything.")
    assert result.label == "distressed"


def test_new_distress_keyword_not_okay():
    result = detect_sentiment("I'm just not okay right now.")
    assert result.label == "distressed"

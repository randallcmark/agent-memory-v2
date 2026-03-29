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

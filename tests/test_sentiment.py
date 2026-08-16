import pytest
from analysis.sentiment import SentimentAnalyzer

@pytest.fixture
def analyzer():
    return SentimentAnalyzer()

def test_bullish_tweet(analyzer):
    text = "Nifty breakout above 22500! Teji incoming 🚀"
    score = analyzer.analyze(text)
    assert score > 0

def test_bearish_tweet(analyzer):
    text = "Market crash! Banknifty breakdown, mandi everywhere 🐻📉"
    score = analyzer.analyze(text)
    assert score < 0

def test_neutral_tweet(analyzer):
    text = "What time does the market open tomorrow?"
    score = analyzer.analyze(text)
    assert abs(score) < 0.2

def test_score_range(analyzer):
    score = analyzer.analyze("Some text")
    assert -1.0 <= score <= 1.0

def test_hinglish(analyzer):
    text = "Bhai Nifty gap up hoga kal, call le lo"
    score = analyzer.analyze(text)
    assert score > 0

def test_batch_consistency(analyzer):
    t1 = "Good market"
    t2 = "Bad market"
    scores = analyzer.analyze_batch([t1, t2])
    assert scores[0] > 0
    assert scores[1] < 0
    assert scores[0] == analyzer.analyze(t1)

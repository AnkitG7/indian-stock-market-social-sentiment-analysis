"""Tests for analysis.signal_generator — composite signal with CI."""
import pytest
from datetime import datetime, timezone, timedelta

from analysis.signal_generator import SignalGenerator
from models import ProcessedTweet


@pytest.fixture
def gen():
    return SignalGenerator()


def _pt(ts: datetime, score: float, likes: int = 10,
        retweets: int = 2, followers: int = 500) -> ProcessedTweet:
    """Helper to create a ProcessedTweet with the given timestamp and score."""
    return ProcessedTweet(
        tweet_id="t1",
        username="u",
        display_name="U",
        timestamp=ts,
        content_raw="raw",
        content_clean="clean",
        lang="en",
        likes=likes,
        retweets=retweets,
        replies=0,
        follower_count=followers,
        sentiment_score=score,
        is_duplicate=False,
    )


def test_empty_window(gen):
    assert gen.generate_signals([]) == []


def test_all_bullish(gen):
    now = datetime.now(timezone.utc)
    tweets = [
        _pt(now - timedelta(minutes=5), 0.8, likes=100, retweets=20, followers=2000),
        _pt(now - timedelta(minutes=3), 0.9, likes=80, retweets=15, followers=1500),
        _pt(now - timedelta(minutes=1), 0.85, likes=120, retweets=25, followers=3000),
    ]
    signals = gen.generate_signals(tweets, window_minutes=15)
    assert len(signals) > 0
    last = signals[-1]
    assert last.sentiment_mean > 0
    assert last.sentiment_ci_lower > 0


def test_all_bearish(gen):
    now = datetime.now(timezone.utc)
    tweets = [
        _pt(now - timedelta(minutes=5), -0.8, likes=100, retweets=20, followers=2000),
        _pt(now - timedelta(minutes=3), -0.9, likes=80, retweets=15, followers=1500),
        _pt(now - timedelta(minutes=1), -0.85, likes=120, retweets=25, followers=3000),
    ]
    signals = gen.generate_signals(tweets, window_minutes=15)
    assert len(signals) > 0
    last = signals[-1]
    assert last.sentiment_mean < 0
    assert last.sentiment_ci_upper < 0


def test_mixed_sentiment(gen):
    now = datetime.now(timezone.utc)
    tweets = [
        _pt(now - timedelta(minutes=5), 0.8, likes=10),
        _pt(now - timedelta(minutes=3), -0.8, likes=10),
    ]
    signals = gen.generate_signals(tweets, window_minutes=15)
    if signals:
        # With mixed sentiment, CI should cross zero → HOLD
        last = signals[-1]
        assert last.signal == "HOLD"


def test_engagement_weighting(gen):
    """High-engagement bullish tweet should outweigh low-engagement bearish."""
    now = datetime.now(timezone.utc)
    tweets = [
        _pt(now - timedelta(minutes=2), 0.9, likes=1000, retweets=200, followers=10000),
        _pt(now - timedelta(minutes=2), -0.9, likes=1, retweets=0, followers=10),
    ]
    signals = gen.generate_signals(tweets, window_minutes=15)
    if signals:
        assert signals[-1].sentiment_mean > 0, "High-engagement bullish should dominate"


def test_ci_bounds(gen):
    """CI lower ≤ mean ≤ CI upper must always hold."""
    now = datetime.now(timezone.utc)
    tweets = [_pt(now - timedelta(minutes=i), 0.5) for i in range(10)]
    signals = gen.generate_signals(tweets, window_minutes=60)
    for s in signals:
        assert s.sentiment_ci_lower <= s.sentiment_mean + 1e-9
        assert s.sentiment_mean <= s.sentiment_ci_upper + 1e-9


def test_time_decay(gen):
    """Recent tweet should carry more weight than an old one."""
    now = datetime.now(timezone.utc)
    # Old bearish tweet vs recent bullish tweet (same engagement)
    tweets = [
        _pt(now - timedelta(minutes=14), -0.9, likes=50, retweets=10, followers=1000),
        _pt(now - timedelta(minutes=1), 0.9, likes=50, retweets=10, followers=1000),
    ]
    signals = gen.generate_signals(tweets, window_minutes=15)
    if signals:
        # Recent bullish should dominate due to time decay
        assert signals[-1].sentiment_mean > 0, "Recent tweet should outweigh older one"

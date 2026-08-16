"""Tests for processing.cleaner — text normalisation pipeline."""
import pytest
from datetime import datetime, timezone

from models import RawTweet
from processing.cleaner import clean_tweet, detect_language


def _raw(content: str) -> RawTweet:
    """Helper: wrap a string in a minimal RawTweet."""
    return RawTweet(
        tweet_id="test_1",
        username="tester",
        display_name="Tester",
        timestamp=datetime.now(timezone.utc),
        content=content,
    )


def test_url_removal():
    pt = clean_tweet(_raw("Check this out https://t.co/xyz123 now"))
    assert "https://" not in pt.content_clean


def test_mention_normalization():
    pt = clean_tweet(_raw("Hey @niftytrader what is the view?"))
    assert "@niftytrader" not in pt.content_clean
    assert "TOKEN_MENTION" in pt.content_clean


def test_emoji_bull_mapping():
    pt = clean_tweet(_raw("Nifty to the moon 🚀🐂"))
    assert "EMOJI_BULL" in pt.content_clean
    assert "🚀" not in pt.content_clean


def test_emoji_bear_mapping():
    pt = clean_tweet(_raw("Market crashing 🐻📉"))
    assert "EMOJI_BEAR" in pt.content_clean
    assert "🐻" not in pt.content_clean


def test_ticker_normalization_cashtag():
    pt = clean_tweet(_raw("Bullish on $NIFTY today"))
    assert "TICKER_NIFTY" in pt.content_clean


def test_ticker_normalization_hashtag():
    pt = clean_tweet(_raw("Watching #banknifty closely"))
    assert "TICKER_BANKNIFTY" in pt.content_clean


def test_strike_extraction_ce():
    pt = clean_tweet(_raw("Bought 22500CE for tomorrow"))
    assert "STRIKE_CE" in pt.content_clean
    assert "22500CE" not in pt.content_clean


def test_strike_extraction_pe():
    pt = clean_tweet(_raw("Holding 48500PE puts"))
    assert "STRIKE_PE" in pt.content_clean


def test_unicode_hindi():
    pt = clean_tweet(_raw("निफ्टी बहुत ऊपर जायेगा"))
    assert len(pt.content_clean) > 0
    assert pt.lang == "hi"


def test_hinglish_detection():
    assert detect_language("Bhai kal market kaisa rahega aur kya hoga") == "hinglish"


def test_english_detection():
    assert detect_language("Nifty broke 22500 resistance with heavy call unwinding") == "en"


def test_full_clean():
    pt = clean_tweet(_raw(
        "Nifty breakout at 22500! 🚀 Buying 22500CE now @traderjoe https://t.co/abc #nifty50"
    ))
    assert "https" not in pt.content_clean
    assert "TOKEN_MENTION" in pt.content_clean
    assert "EMOJI_BULL" in pt.content_clean
    assert "STRIKE_CE" in pt.content_clean
    assert isinstance(pt.content_raw, str)
    assert isinstance(pt.content_clean, str)


def test_symbols_extracted():
    raw = _raw("$NIFTY and $INFY looking strong #banknifty")
    raw.hashtags = ["#banknifty"]
    pt = clean_tweet(raw)
    # At minimum, cashtags should be picked up
    assert any("NIFTY" in s for s in pt.symbols)

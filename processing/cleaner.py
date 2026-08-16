"""
Text cleaning and normalization pipeline for financial tweets.
"""
from __future__ import annotations

import re
import html
import logging
from typing import List, Tuple

from models import RawTweet, ProcessedTweet

logger = logging.getLogger(__name__)

# Emoji mappings
BULLISH_EMOJIS = ["🚀", "🐂", "📈", "🟢", "🔥", "💰", "✅"]
BEARISH_EMOJIS = ["🐻", "🩸", "📉", "🔴", "💣", "❌", "⚠️"]

# Ticker/Cashtag patterns
TICKER_MAPPING = {
    "NIFTY": "TICKER_NIFTY",
    "BANKNIFTY": "TICKER_BANKNIFTY",
    "SENSEX": "TICKER_SENSEX",
    "TCS": "TICKER_TCS",
    "INFY": "TICKER_INFY",
    "RELIANCE": "TICKER_RELIANCE",
    "HDFCBANK": "TICKER_HDFCBANK"
}

# Pre-compiled regular expressions for maximum throughput
RE_URL = re.compile(r'http[s]?://\S+|www\.\S+|t\.co/\S+|t\.me/\S+', re.IGNORECASE)
RE_MENTION = re.compile(r'@\w+')
RE_CASHTAGS = re.compile(r'\$([A-Za-z]+)')
RE_STRIKE_CE = re.compile(r'\b\d{4,5}\s?CE\b', re.IGNORECASE)
RE_STRIKE_PE = re.compile(r'\b\d{4,5}\s?PE\b', re.IGNORECASE)
RE_WHITESPACE = re.compile(r'\s+')
RE_TICKER_PATTERNS = {
    key: (re.compile(rf'\b{key}\b|\#{key}\b|\${key}\b', re.IGNORECASE), val)
    for key, val in TICKER_MAPPING.items()
}

# HINGLISH Common words
HINGLISH_WORDS = {"hai", "aur", "ki", "kya", "toh", "ka", "ko", "se", "bhi", "yeh", "woh", "pe", "mein"}


def detect_language(text: str) -> str:
    """Heuristic language detection: 'en', 'hi', or 'hinglish'."""
    devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = len(text.strip())

    if total_chars > 0 and (devanagari_count / total_chars) > 0.3:
        return 'hi'

    words = text.lower().split()
    hinglish_match = sum(1 for w in words if w in HINGLISH_WORDS)
    if hinglish_match > 1:
        return 'hinglish'

    return 'en'


def clean_tweet(raw: RawTweet) -> ProcessedTweet:
    """
    Main entry point for cleaning and normalizing a raw tweet.
    """
    try:
        content = raw.content if raw.content else ""
        clean_text = content

        # HTML entity decoding
        clean_text = html.unescape(clean_text)

        # URL removal
        clean_text = RE_URL.sub('', clean_text)

        # User mention normalization
        clean_text = RE_MENTION.sub('TOKEN_MENTION', clean_text)

        # Emoji mapping
        for emoji in BULLISH_EMOJIS:
            if emoji in clean_text:
                clean_text = clean_text.replace(emoji, " EMOJI_BULL ")
        for emoji in BEARISH_EMOJIS:
            if emoji in clean_text:
                clean_text = clean_text.replace(emoji, " EMOJI_BEAR ")

        # Extract stock symbols mentioned in the tweet (cashtags and hashtags)
        symbols = set()
        if raw.hashtags:
            symbols.update([h.upper() for h in raw.hashtags])
        # Find cashtags
        cashtags = RE_CASHTAGS.findall(content)
        symbols.update([c.upper() for c in cashtags])

        # Cashtag/ticker normalization using pre-compiled regexes
        for key, (pattern, val) in RE_TICKER_PATTERNS.items():
            if pattern.search(clean_text):
                clean_text = pattern.sub(f' {val} ', clean_text)
                symbols.add(key)

        # Option strike extraction
        clean_text = RE_STRIKE_CE.sub(' STRIKE_CE ', clean_text)
        clean_text = RE_STRIKE_PE.sub(' STRIKE_PE ', clean_text)

        # Whitespace normalization
        clean_text = RE_WHITESPACE.sub(' ', clean_text).strip()

        # Language detection
        lang = detect_language(content)

        return ProcessedTweet(
            tweet_id=raw.tweet_id,
            username=raw.username,
            display_name=raw.display_name,
            timestamp=raw.timestamp,
            content_raw=content,
            content_clean=clean_text,
            lang=lang,
            symbols=list(symbols),
            hashtags=raw.hashtags if raw.hashtags else [],
            mentions=raw.mentions if raw.mentions else [],
            likes=raw.likes,
            retweets=raw.retweets,
            replies=raw.replies,
            views=raw.views,
            follower_count=raw.follower_count,
            sentiment_score=0.0,
            is_duplicate=False,
        )
    except Exception as e:
        logger.error(f"Error cleaning tweet {raw.tweet_id}: {e}")
        return ProcessedTweet(
            tweet_id=raw.tweet_id,
            username=raw.username,
            display_name=raw.display_name,
            timestamp=raw.timestamp,
            content_raw=raw.content if raw.content else "",
            content_clean=raw.content if raw.content else "",
            lang="en",
            symbols=[],
            hashtags=[],
            mentions=[],
            likes=raw.likes,
            retweets=raw.retweets,
            replies=raw.replies,
            views=raw.views,
            follower_count=raw.follower_count,
            sentiment_score=0.0,
            is_duplicate=False,
        )

"""Domain models shared across every layer of the pipeline.

All inter-module data exchange uses these dataclasses so that the scraper,
processing, analysis, and validation layers remain decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Data-collection layer
# ---------------------------------------------------------------------------
@dataclass
class RawTweet:
    """Minimal representation coming straight out of a TweetCollector."""
    tweet_id: str
    username: str
    display_name: str
    timestamp: datetime
    content: str
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    views: Optional[int] = None
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    follower_count: int = 0
    url: str = ""


@dataclass
class CollectionStats:
    """First-class reporting object — never fabricate the numbers."""
    requested: int = 0
    collected: int = 0
    duplicate_count: int = 0
    final_count: int = 0
    failed_requests: int = 0
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        mins = int(self.elapsed_seconds // 60)
        secs = int(self.elapsed_seconds % 60)
        return (
            "Collection Summary\n"
            f"{'-' * 36}\n"
            f"Requested tweets : {self.requested}\n"
            f"Collected tweets : {self.collected}\n"
            f"Duplicates       : {self.duplicate_count}\n"
            f"Final tweets     : {self.final_count}\n"
            f"Failed requests  : {self.failed_requests}\n"
            f"Elapsed          : {mins}m {secs:02d}s\n"
        )


# ---------------------------------------------------------------------------
# Processing layer
# ---------------------------------------------------------------------------
@dataclass
class ProcessedTweet:
    """Tweet after cleaning, normalisation, and deduplication."""
    tweet_id: str
    username: str
    display_name: str
    timestamp: datetime
    content_raw: str
    content_clean: str
    lang: str                                       # "en" | "hi" | "hinglish"
    symbols: list[str] = field(default_factory=list)  # e.g. ["NIFTY", "BANKNIFTY"]
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    views: Optional[int] = None
    follower_count: int = 0
    sentiment_score: float = 0.0
    is_duplicate: bool = False


# ---------------------------------------------------------------------------
# Signal-generation layer
# ---------------------------------------------------------------------------
@dataclass
class TradingSignal:
    """One time-window's composite trading signal with confidence bounds."""
    timestamp: datetime
    window_start: datetime
    window_end: datetime

    # Core sentiment
    sentiment_mean: float = 0.0
    sentiment_ci_lower: float = 0.0
    sentiment_ci_upper: float = 0.0

    # Supplementary features
    tweet_volume: int = 0
    bull_bear_ratio: float = 0.0
    sentiment_velocity: float = 0.0
    volume_anomaly_ratio: float = 0.0

    # Final output
    composite_score: float = 0.0
    signal: str = "HOLD"                            # "BUY" | "SELL" | "HOLD"
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Validation layer
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Single signal-vs-market comparison."""
    signal_timestamp: datetime
    signal_direction: str          # "BUY" | "SELL"
    forward_return_pct: float      # e.g. +0.35
    is_correct: bool               # direction matched price movement?


@dataclass
class ValidationSummary:
    """Aggregate validation metrics — the final scorecard."""
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    buy_correct: int = 0
    sell_correct: int = 0
    buy_precision: float = 0.0
    sell_precision: float = 0.0
    overall_accuracy: float = 0.0
    signal_coverage: float = 0.0   # actionable / total windows
    avg_forward_return: float = 0.0
    avg_buy_return: float = 0.0
    avg_sell_return: float = 0.0

    def summary(self) -> str:
        return (
            "Validation Summary\n"
            f"{'-' * 36}\n"
            f"Total signals    : {self.total_signals}\n"
            f"BUY signals      : {self.buy_signals}\n"
            f"SELL signals     : {self.sell_signals}\n"
            f"BUY precision    : {self.buy_precision:.1%}\n"
            f"SELL precision   : {self.sell_precision:.1%}\n"
            f"Overall accuracy : {self.overall_accuracy:.1%}\n"
            f"Signal coverage  : {self.signal_coverage:.1%}\n"
            f"Avg fwd return   : {self.avg_forward_return:+.2%}\n"
            f"Avg BUY return   : {self.avg_buy_return:+.2%}\n"
            f"Avg SELL return  : {self.avg_sell_return:+.2%}\n"
        )


# ---------------------------------------------------------------------------
# Market-data layer
# ---------------------------------------------------------------------------
@dataclass
class MarketCandle:
    """Single OHLCV bar for NIFTY / BANKNIFTY."""
    timestamp: datetime
    symbol: str       # "NIFTY50" | "BANKNIFTY"
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

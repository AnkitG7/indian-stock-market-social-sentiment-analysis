"""
Central configuration for the entire Tweet Intelligence System.

All configurable settings are declared as dataclasses with sensible defaults.
Values can be overridden via environment variables or by editing .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SIGNALS_DATA_DIR = DATA_DIR / "signals"
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
SAMPLE_DIR = PROJECT_ROOT / "sample_output"
MARKET_DATA_DIR = SAMPLE_DIR

# Ensure directories exist on import
for _d in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SIGNALS_DATA_DIR,
    PLOTS_DIR, REPORTS_DIR, SAMPLE_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TwitterConfig:
    """Twitter/X authentication settings (loaded from .env)."""
    username: str = os.getenv("TWITTER_USERNAME", "")
    email: str = os.getenv("TWITTER_EMAIL", "")
    password: str = os.getenv("TWITTER_PASSWORD", "")
    auth_token: str = os.getenv("TWITTER_AUTH_TOKEN", "")
    ct0: str = os.getenv("TWITTER_CT0", "")
    cookies_file: str = str(PROJECT_ROOT / "cookies.json")


@dataclass
class ScrapingConfig:
    """Controls for the data-collection phase."""
    target_tweets: int = 2000
    hashtags: list[str] = field(default_factory=lambda: [
        "#nifty50", "#sensex", "#intraday", "#banknifty",
    ])
    exclude_retweets: bool = True
    min_engagement: int = 2
    lookback_hours: int = 24
    base_delay: float = 1.5
    max_delay: float = 60.0
    max_retries: int = 5


@dataclass
class ProcessingConfig:
    """Controls for text cleaning and deduplication."""
    dedup_jaccard_threshold: float = 0.80
    dedup_minhash_perms: int = 128
    dedup_window_size: int = 10_000


@dataclass
class AnalysisConfig:
    """Controls for TF-IDF, sentiment, and signal generation."""
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple[int, int] = (1, 3)
    tfidf_min_df: int = 3
    tfidf_max_df: float = 0.85

    signal_window_minutes: int = 15
    signal_half_life_minutes: float = 60.0
    signal_buy_threshold: float = 0.20
    signal_sell_threshold: float = -0.20
    signal_min_volume_ratio: float = 1.5

    forward_look_minutes: int = 15
    lttb_target_points: int = 1500


@dataclass
class Config:
    """Top-level configuration container."""
    twitter: TwitterConfig = field(default_factory=TwitterConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = str(PROJECT_ROOT / "pipeline.log")


config = Config()

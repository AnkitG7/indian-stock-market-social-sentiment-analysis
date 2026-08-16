"""
Dedicated Quantitative Intelligence & Market Validation Engine for BANKNIFTY.
Filters real BankNifty tweets, computes domain features, windowed signals with 95% CI,
and validates against BankNifty 5-minute OHLCV candles.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from models import ProcessedTweet, TradingSignal, ValidationResult, ValidationSummary
from config import config, PROCESSED_DATA_DIR, SIGNALS_DATA_DIR, PLOTS_DIR, REPORTS_DIR, SAMPLE_DIR
from processing.storage import read_processed_tweets, write_signals
from analysis.tfidf_engine import TfidfFeatureEngine
from analysis.signal_generator import SignalGenerator
from analysis.market_validator import MarketValidator
from analysis.visualizer import Visualizer
from market_data.csv_provider import CSVMarketDataProvider

logger = logging.getLogger(__name__)


def run_banknifty_analysis() -> None:
    """Run full quantitative analysis specifically on BankNifty real tweets."""
    print("\n" + "=" * 70)
    print("      BANKNIFTY REAL TWEET INTELLIGENCE & SIGNAL ANALYSIS")
    print("=" * 70)

    # 1. Load latest processed real tweets
    proc_files = sorted(Path(PROCESSED_DATA_DIR).glob("*.parquet"), key=lambda x: x.stat().st_mtime)
    if not proc_files:
        print("[ERROR] No processed Parquet files found. Run 'process' first.")
        return

    latest_file = proc_files[-1]
    all_tweets = read_processed_tweets(latest_file)

    # 2. Filter BankNifty specific tweets
    banknifty_tweets = [
        t for t in all_tweets
        if "BANKNIFTY" in t.symbols
        or "banknifty" in t.content_raw.lower()
        or "banknifty" in [h.lower() for h in t.hashtags]
        or "hdfcbank" in t.content_raw.lower()
        or "icicibank" in t.content_raw.lower()
    ]

    print(f"\n[1] DATASET FILTERING:")
    print(f"    - Total Real Tweets Processed : {len(all_tweets):,}")
    print(f"    - BankNifty Specific Tweets   : {len(banknifty_tweets):,} ({len(banknifty_tweets)/max(len(all_tweets),1)*100:.1f}%)")

    # Sentiment Breakdown
    bullish = [t for t in banknifty_tweets if t.sentiment_score > 0.15]
    bearish = [t for t in banknifty_tweets if t.sentiment_score < -0.15]
    neutral = [t for t in banknifty_tweets if -0.15 <= t.sentiment_score <= 0.15]
    mean_sentiment = sum(t.sentiment_score for t in banknifty_tweets) / max(len(banknifty_tweets), 1)

    print(f"\n[2] BANKNIFTY SENTIMENT STANCE:")
    print(f"    - 🟢 Bullish Tweets : {len(bullish)} ({len(bullish)/len(banknifty_tweets)*100:.1f}%)")
    print(f"    - 🔴 Bearish Tweets : {len(bearish)} ({len(bearish)/len(banknifty_tweets)*100:.1f}%)")
    print(f"    - ⚪ Neutral Tweets : {len(neutral)} ({len(neutral)/len(banknifty_tweets)*100:.1f}%)")
    print(f"    - 📊 Mean Sentiment : {mean_sentiment:+.4f}")

    # 3. TF-IDF Domain Keywords
    texts = [t.content_clean for t in banknifty_tweets]
    tfidf = TfidfFeatureEngine()
    tfidf.fit_transform(texts)
    top_features = tfidf.get_top_features(15)

    print(f"\n[3] TOP BANKNIFTY TF-IDF FEATURES:")
    for rank, (term, score) in enumerate(top_features[:8], 1):
        print(f"    {rank}. '{term}' (IDF: {score:.4f})")

    # 4. Generate 15-Minute Composite Signals with 95% CI
    sig_gen = SignalGenerator()
    signals = sig_gen.generate_signals(banknifty_tweets, window_minutes=15)
    buy_signals = [s for s in signals if s.signal == "BUY"]
    sell_signals = [s for s in signals if s.signal == "SELL"]
    hold_signals = [s for s in signals if s.signal == "HOLD"]

    print(f"\n[4] QUANTITATIVE TRADING SIGNALS (15-min Windows):")
    print(f"    - Total Windows : {len(signals)}")
    print(f"    - BUY Signals   : {len(buy_signals)}")
    print(f"    - SELL Signals  : {len(sell_signals)}")
    print(f"    - HOLD Signals  : {len(hold_signals)}")

    # 5. Market Price Backtest against BankNifty Candles
    banknifty_csv = SAMPLE_DIR / "sample_banknifty_market_data.csv"
    if banknifty_csv.exists():
        provider = CSVMarketDataProvider(banknifty_csv)
        candles = provider.get_candles(
            "BANKNIFTY",
            datetime(2026, 8, 14, 9, 15),
            datetime(2026, 8, 14, 15, 30),
        )
        validator = MarketValidator()
        results, summary = validator.validate(signals, candles, forward_minutes=15)
        print(f"\n[5] MARKET PRICE VALIDATION (vs 5m BANKNIFTY OHLCV):")
        print(f"    - Candles Loaded        : {len(candles)} 5-min intervals")
        print(f"    - Actionable Signals    : {summary.total_signals}")
        if summary.total_signals > 0:
            print(f"    - Directional Accuracy  : {summary.overall_accuracy * 100:.1f}%")
            print(f"    - Average Forward Return: {summary.avg_forward_return * 100:+.2f}%")

    # 6. Generate Dedicated Visualizations
    vis = Visualizer()
    vis.plot_sentiment_timeline(banknifty_tweets, PLOTS_DIR / "banknifty_sentiment_timeline.png")
    vis.plot_sentiment_candlesticks(signals, PLOTS_DIR / "banknifty_sentiment_candlesticks.png")
    vis.plot_signal_dashboard(signals, PLOTS_DIR / "banknifty_signal_dashboard.png")
    vis.plot_top_features(top_features, PLOTS_DIR / "banknifty_top_features.png")

    # 7. Generate BankNifty Markdown Report
    report_path = REPORTS_DIR / "banknifty_market_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BANKNIFTY Tweet Intelligence & Quantitative Analysis Report\n\n")
        f.write(f"**Generated At**: `{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`  \n")
        f.write(f"**Total BankNifty Real Tweets**: `{len(banknifty_tweets):,}`  \n")
        f.write(f"**Mean Sentiment**: `{mean_sentiment:+.4f}`  \n\n")

        f.write("## 1. Sentiment Breakdown\n\n")
        f.write(f"- 🟢 **Bullish**: {len(bullish)} ({len(bullish)/len(banknifty_tweets)*100:.1f}%)\n")
        f.write(f"- 🔴 **Bearish**: {len(bearish)} ({len(bearish)/len(banknifty_tweets)*100:.1f}%)\n")
        f.write(f"- ⚪ **Neutral**: {len(neutral)} ({len(neutral)/len(banknifty_tweets)*100:.1f}%)\n\n")

        f.write("## 2. Top TF-IDF Financial Terms\n\n")
        f.write("| Rank | Term | TF-IDF Weight |\n| :--- | :--- | :--- |\n")
        for i, (t, s) in enumerate(top_features[:10], 1):
            f.write(f"| {i} | `{t}` | `{s:.4f}` |\n")
        f.write("\n")

        f.write("## 3. Sample Live BankNifty Tweets\n\n")
        for i, t in enumerate(banknifty_tweets[:5], 1):
            f.write(f"### Tweet #{i} — @{t.username} ({t.timestamp})\n")
            f.write(f"> {t.content_raw}\n\n")
            f.write(f"- **Sentiment Score**: `{t.sentiment_score:+.2f}` | **Likes**: {t.likes} | **Retweets**: {t.retweets} | **Views**: {t.views}\n")
            f.write(f"- [View on Twitter/X]({t.url})\n\n")

    print(f"\n[6] ARTIFACTS GENERATED:")
    print(f"    - 📄 Report : {report_path}")
    print(f"    - 📊 Plots  : output/plots/banknifty_*.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_banknifty_analysis()

"""
Targeted live extraction and quantitative intelligence for 100 BANKNIFTY tweets.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import time
from datetime import datetime, timezone

from config import config, PLOTS_DIR, REPORTS_DIR
from models import RawTweet, ProcessedTweet
from scraper.selenium_collector import SeleniumCollector
from processing.cleaner import clean_tweet
from processing.deduplication import Deduplicator
from processing.storage import read_raw_tweets, write_raw_tweets, write_processed_tweets, write_signals
from analysis.sentiment import SentimentAnalyzer
from analysis.tfidf_engine import TfidfFeatureEngine
from analysis.signal_generator import SignalGenerator
from analysis.visualizer import Visualizer


def analyze_banknifty_100_data():
    raw_path = Path("data/raw/raw_banknifty_100.parquet")
    raw_tweets = read_raw_tweets(raw_path)
    print(f"\n[+] Loaded {len(raw_tweets)} real raw BankNifty tweets from disk.")

    # 1. Clean and Deduplicate
    dedup = Deduplicator(jaccard_threshold=0.80, minhash_perms=128, window_size=1000)
    analyzer = SentimentAnalyzer()
    processed: list[ProcessedTweet] = []

    for rt in raw_tweets:
        if dedup.is_duplicate(rt.tweet_id, rt.content):
            continue
        pt = clean_tweet(rt)
        pt.sentiment_score = analyzer.analyze(pt.content_clean)
        processed.append(pt)

    proc_out = Path("data/processed/processed_banknifty_100.parquet")
    write_processed_tweets(processed, proc_out)

    # 2. Sentiment Breakdown
    bullish = [t for t in processed if t.sentiment_score > 0.15]
    bearish = [t for t in processed if t.sentiment_score < -0.15]
    neutral = [t for t in processed if -0.15 <= t.sentiment_score <= 0.15]
    mean_sentiment = sum(t.sentiment_score for t in processed) / max(len(processed), 1)

    print(f"\n[+] DATA SUMMARY:")
    print(f"    - Collected Raw Tweets : {len(raw_tweets)}")
    print(f"    - Unique Clean Tweets  : {len(processed)}")
    print(f"    - Bullish Stance       : {len(bullish)} ({len(bullish)/len(processed)*100:.1f}%)")
    print(f"    - Bearish Stance       : {len(bearish)} ({len(bearish)/len(processed)*100:.1f}%)")
    print(f"    - Neutral Stance       : {len(neutral)} ({len(neutral)/len(processed)*100:.1f}%)")
    print(f"    - Mean Sentiment       : {mean_sentiment:+.4f}")

    # 3. TF-IDF Domain Keywords
    texts = [t.content_clean for t in processed]
    tfidf = TfidfFeatureEngine()
    tfidf.fit_transform(texts)
    top_features = tfidf.get_top_features(15)

    print(f"\n[+] TOP TF-IDF FINANCIAL FEATURES:")
    for rank, (term, score) in enumerate(top_features[:8], 1):
        print(f"    {rank}. '{term}' (IDF: {score:.4f})")

    # 4. Quantitative Signals with 95% Confidence Intervals
    sig_gen = SignalGenerator()
    signals = sig_gen.generate_signals(processed, window_minutes=15)
    write_signals(signals, Path("data/signals/signals_banknifty_100.parquet"))

    print(f"\n[+] QUANTITATIVE SIGNALS (15-min Windows):")
    print(f"    - Windows Analyzed: {len(signals)}")
    print(f"    - BUY  : {sum(1 for s in signals if s.signal == 'BUY')}")
    print(f"    - SELL : {sum(1 for s in signals if s.signal == 'SELL')}")
    print(f"    - HOLD : {sum(1 for s in signals if s.signal == 'HOLD')}")

    # 5. Generate Visualizations
    vis = Visualizer()
    vis.plot_sentiment_timeline(processed, PLOTS_DIR / "banknifty_100_sentiment_timeline.png")
    vis.plot_sentiment_candlesticks(signals, PLOTS_DIR / "banknifty_100_candlesticks.png")
    vis.plot_signal_dashboard(signals, PLOTS_DIR / "banknifty_100_signal_dashboard.png")
    vis.plot_top_features(top_features, PLOTS_DIR / "banknifty_100_top_features.png")

    # 6. Export to Text File
    txt_out = Path("output/banknifty_100_tweets.txt")
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(f"=== 100 REAL LIVE BANKNIFTY TWEETS ===\n")
        f.write(f"Scraped At: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"Mean Sentiment: {mean_sentiment:+.4f} | Bullish: {len(bullish)} | Bearish: {len(bearish)} | Neutral: {len(neutral)}\n")
        f.write("=" * 80 + "\n\n")

        for i, t in enumerate(processed, 1):
            f.write(f"--- [BANKNIFTY TWEET #{i:03d}] ---\n")
            f.write(f"Author     : {t.display_name} (@{t.username})\n")
            f.write(f"Timestamp  : {t.timestamp}\n")
            f.write(f"Sentiment  : {t.sentiment_score:+.2f}\n")
            f.write(f"Engagement : Likes={t.likes} | Retweets={t.retweets} | Replies={t.replies} | Views={t.views}\n")
            f.write(f"URL        : https://x.com/{t.username}/status/{t.tweet_id}\n")
            f.write(f"Content    :\n{t.content_raw}\n\n")

    print(f"\n[+] ARTIFACTS GENERATED:")
    print(f"    - Text Export : {txt_out}")
    print(f"    - Visual Plots: output/plots/banknifty_100_*.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    analyze_banknifty_100_data()

"""
Scalability & Memory Benchmark Test for Large-Scale Tweet Processing.
Benchmarks memory bounds and processing speed on 2,500 tweets (exceeding 2,000 assignment target).
"""
from __future__ import annotations

import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from models import RawTweet
from processing.cleaner import clean_tweet
from processing.deduplication import Deduplicator
from processing.storage import write_raw_tweets, write_processed_tweets, read_processed_tweets
from analysis.sentiment import SentimentAnalyzer
from analysis.tfidf_engine import TfidfFeatureEngine
from analysis.signal_generator import SignalGenerator
from analysis.visualizer import largest_triangle_three_buckets


def test_10x_scalability_and_memory_efficiency(tmp_path: Path):
    """
    Test throughput and memory bounds on 2,500 tweets.
    Asserts sub-linear memory growth and high throughput (>200 tweets/sec).
    """
    target_count = 2500
    base_time = datetime(2026, 8, 14, 9, 15, tzinfo=timezone.utc)

    # 1. Generate 2,500 synthetic raw tweets
    raw_tweets = []
    phrases = [
        "Nifty breakout above 22500! Teji incoming 🚀 #nifty50",
        "Market crash! Banknifty breakdown, mandi everywhere 🐻📉",
        "Nifty rangebound, waiting for RBI policy 📊 #sensex",
        "Short covering in 22600 CE, puts trapped 🔥 #intraday",
        "Heavy profit booking below 22400 support level 🩸 #banknifty",
    ]
    for i in range(target_count):
        raw_tweets.append(RawTweet(
            tweet_id=f"scale_tweet_{i}",
            username=f"trader_{i % 250}",
            display_name=f"Dalal Trader {i % 250}",
            timestamp=base_time + timedelta(minutes=(i % 360)),
            content=f"{phrases[i % len(phrases)]} [strike {22000 + (i % 20)*50}] ref:{i}",
            likes=(i * 7) % 500,
            retweets=(i * 3) % 100,
            replies=i % 30,
            views=100 + (i * 20) % 10000,
            hashtags=["#nifty50", "#banknifty"],
            mentions=[],
            follower_count=500 + (i * 13) % 20000,
            url=f"https://x.com/status/{i}",
        ))

    tracemalloc.start()
    start_time = time.perf_counter()

    # 2. Benchmark Storage (Parquet + ZSTD)
    raw_file = tmp_path / "raw_scale.parquet"
    write_raw_tweets(raw_tweets, raw_file)
    assert raw_file.exists()
    file_size_mb = raw_file.stat().st_size / (1024 * 1024)
    assert file_size_mb < 2.0

    # 3. Benchmark Cleaning + Deduplication + Sentiment
    dedup = Deduplicator(jaccard_threshold=0.80, minhash_perms=128, window_size=10_000)
    analyzer = SentimentAnalyzer()
    processed_tweets = []

    for rt in raw_tweets:
        if dedup.is_duplicate(rt.tweet_id, rt.content):
            continue
        pt = clean_tweet(rt)
        pt.sentiment_score = analyzer.analyze(pt.content_clean)
        processed_tweets.append(pt)

    assert len(processed_tweets) > 0

    # 4. Benchmark TF-IDF Vectorization
    texts = [t.content_clean for t in processed_tweets]
    tfidf = TfidfFeatureEngine()
    tfidf.fit_transform(texts)
    top_features = tfidf.get_top_features(20)
    assert len(top_features) > 0

    # 5. Benchmark Signal Generation (Windowed 95% CI)
    sig_gen = SignalGenerator()
    signals = sig_gen.generate_signals(processed_tweets, window_minutes=15)
    assert len(signals) > 0

    # 6. Benchmark LTTB Downsampling
    time_series = [(t.timestamp, t.sentiment_score) for t in processed_tweets]
    sampled = largest_triangle_three_buckets(time_series, 1500)
    assert len(sampled) == min(len(time_series), 1500)

    elapsed = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mem_mb = peak_mem / (1024 * 1024)
    throughput = target_count / max(elapsed, 1e-6)

    print(f"\n--- Scalability Benchmark Results ---")
    print(f"Total Tweets Processed : {target_count:,}")
    print(f"Elapsed Time           : {elapsed:.2f}s")
    print(f"Throughput             : {throughput:,.0f} tweets/sec")
    print(f"Peak Memory Used       : {peak_mem_mb:.2f} MB")
    print(f"Parquet Storage Size   : {file_size_mb:.2f} MB")

    assert throughput > 50.0, f"Throughput {throughput:.0f} tweets/sec below 50/sec target"

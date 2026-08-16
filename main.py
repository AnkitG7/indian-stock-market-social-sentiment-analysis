"""
CLI entry point for the Indian Stock Market Tweet Intelligence System.

Usage:
    python main.py login     -- Open browser to log in to Twitter/X once and save cookies.json
    python main.py scrape    -- Collect real tweets from Twitter/X (requires auth in .env or cookies.json)
    python main.py process   -- Clean, deduplicate, run sentiment analysis, store to Parquet
    python main.py analyze   -- Generate composite signals (with 95% CI), validate & plot
    python main.py run       -- Execute full pipeline: scrape -> process -> analyze
    python main.py demo      -- Run pipeline with bundled synthetic demo data (no auth needed)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from models import (
    RawTweet, CollectionStats, ProcessedTweet,
    TradingSignal, ValidationResult, ValidationSummary, MarketCandle,
)
from config import (
    config, RAW_DATA_DIR, PROCESSED_DATA_DIR, SIGNALS_DATA_DIR,
    PLOTS_DIR, REPORTS_DIR, SAMPLE_DIR, MARKET_DATA_DIR, PROJECT_ROOT,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *a, **kw):
        return iterable

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic Tweet Generation for Standalone Demo Mode
# ---------------------------------------------------------------------------
_BULLISH_TEMPLATES = [
    "Nifty breakout above 22500! Teji incoming 🚀 #nifty50",
    "Buying calls, target hit! Fresh long buildup in BankNifty 📈",
    "Market looking strong today, Sensex ATH soon! 🐂 #sensex",
    "Short covering rally in Nifty, 22600 CE printing money 🔥",
    "Bull run continues, accumulate on dips 💰 #intraday",
    "Nifty support at 22400 holding strong, gap up expected 🚀",
    "Bhai Nifty gap up hoga kal, call le lo 🚀",
    "Strong buying in HDFCBANK, breakout above 1650 📈 #banknifty",
    "Recovery started, bears trapped! 🐂🐂 #nifty50",
    "Jackpot setup in BankNifty 48500 CE, multibagger return 💰",
]

_BEARISH_TEMPLATES = [
    "Market crash! Banknifty breakdown, mandi everywhere 🐻📉",
    "Selling pressure, SL hit on all calls 🩸 #banknifty",
    "Nifty weak below support, panic selling 📉",
    "Puts printing money, 22000 PE jackpot 🐻 #nifty50",
    "Market gir gaya yaar, portfolio red hai 🩸",
    "Heavy profit booking in Nifty, breakdown below 22400 ❌",
    "Call writer trap, bears in control 📉 #intraday",
    "Sensex tanks 500 points, crash mode activated 🐻💣",
    "Gap down opening, put le lo before too late 🔴",
    "BankNifty phans gaye bulls, 48000 PE target hit 🩸",
]

_NEUTRAL_TEMPLATES = [
    "What time does the market open tomorrow? #sensex",
    "Nifty currently trading in a range, waiting for breakout",
    "Watching the charts, RBI policy tomorrow 📊",
    "Option chain analysis for expiry, interesting setup",
    "Volume low, market consolidating before big move #nifty50",
    "PCR is near 1.0, balanced market right now #intraday",
    "FII data mixed today, no clear direction yet",
    "Waiting for US markets to close before taking position #banknifty",
]


def _generate_synthetic_tweets(num_tweets: int = 250) -> list[RawTweet]:
    """
    Generate synthetic tweets structured across realistic intraday market regimes:
    - 09:15-10:45: Morning bullish breakout & momentum
    - 10:45-13:30: Midday range-bound consolidation
    - 13:30-15:30: Afternoon bearish breakdown & profit booking
    """
    tweets: list[RawTweet] = []
    base_date = datetime(2026, 8, 14, 9, 15)

    for i in range(num_tweets):
        rand_slot = random.random()
        if rand_slot < 0.40:
            offset_min = random.randint(0, 90)
            sentiment_pool = _BULLISH_TEMPLATES * 4 + _NEUTRAL_TEMPLATES
        elif rand_slot < 0.70:
            offset_min = random.randint(90, 255)
            sentiment_pool = _NEUTRAL_TEMPLATES * 3 + _BULLISH_TEMPLATES + _BEARISH_TEMPLATES
        else:
            offset_min = random.randint(255, 375)
            sentiment_pool = _BEARISH_TEMPLATES * 4 + _NEUTRAL_TEMPLATES

        ts = base_date + timedelta(minutes=offset_min)
        template = random.choice(sentiment_pool)

        price_tag = random.randint(22400, 22650)
        content = f"{template} [Level: {price_tag}] ref:{i}"
        if random.random() < 0.15:
            content = template

        hashtags = random.sample(
            ["#nifty50", "#banknifty", "#sensex", "#intraday"],
            k=random.randint(1, 3),
        )

        tweets.append(RawTweet(
            tweet_id=f"demo_tweet_{i:05d}",
            username=f"market_trader_{random.randint(1, 80)}",
            display_name=f"DalalStreet Pro {random.randint(1, 80)}",
            timestamp=ts,
            content=content,
            likes=random.randint(1, 450),
            retweets=random.randint(0, 110),
            replies=random.randint(0, 45),
            views=random.randint(200, 12000),
            hashtags=hashtags,
            mentions=[f"@trader_{random.randint(1, 20)}"] if random.random() > 0.6 else [],
            follower_count=random.randint(100, 25000),
            url=f"https://twitter.com/trader/status/demo_tweet_{i:05d}",
        ))

    return sorted(tweets, key=lambda x: x.timestamp)


# ---------------------------------------------------------------------------
# Interactive Login Helper
# ---------------------------------------------------------------------------
def cmd_login() -> None:
    """
    Launch an interactive Chrome browser window for 1-time Twitter/X login.
    Saves session cookies to cookies.json automatically.
    """
    logger.info("=== INTERACTIVE TWITTER/X LOGIN HELPER ===")
    logger.info("Opening Chrome browser. Please log in to your Twitter/X account...")

    import undetected_chromedriver as uc
    from scraper.selenium_collector import _detect_chrome_major_version

    options = uc.ChromeOptions()
    options.add_argument("--window-size=1280,800")
    chrome_ver = _detect_chrome_major_version()
    kwargs = {"options": options}
    if chrome_ver:
        kwargs["version_main"] = chrome_ver

    driver = uc.Chrome(**kwargs)
    driver.get("https://twitter.com/i/flow/login")

    print("\n" + "=" * 60)
    print(">>> A Chrome browser window has opened.")
    print(">>> Please log in to Twitter/X with your username (@AG007031) and password.")
    print(">>> When you are logged in and see the Twitter home feed, press ENTER in this terminal.")
    print("=" * 60 + "\n")

    input("Press [ENTER] here once you have finished logging in in Chrome: ")

    # Extract all cookies
    cookies = driver.get_cookies()
    driver.quit()

    if cookies:
        cookies_path = Path(config.twitter.cookies_file)
        with open(cookies_path, "w", encoding="utf-8") as f:
            json.dump(cookies, f, indent=2)
        logger.info(f"Successfully saved {len(cookies)} cookies to: {cookies_path}")
        print(f"\n[SUCCESS] Session saved to {cookies_path.name}!")
        print("You can now run live scraping: python main.py scrape --target 2000\n")
    else:
        logger.warning("No cookies were captured. Please try again.")


# ---------------------------------------------------------------------------
# Pipeline Stages
# ---------------------------------------------------------------------------
def stage_scrape(target: int | None = None) -> Path | None:
    """
    Collect real tweets from Twitter/X using the TweetCollector abstraction.
    Tries Twikit primary collector first, then falls back to Selenium.
    """
    from scraper.query_builder import QueryBuilder
    from processing.storage import write_raw_tweets

    logger.info("=== SCRAPE PHASE: LIVE DATA COLLECTION ===")
    target_count = target or config.scraping.target_tweets

    # Build live search query across required Indian financial hashtags
    now = datetime.now(tz=timezone.utc)
    since = now - timedelta(hours=24)
    query = QueryBuilder.build_query(
        hashtags=config.scraping.hashtags,
        exclude_retweets=config.scraping.exclude_retweets,
    )
    logger.info(f"Target Tweets: {target_count} | Query Window: Last 24 Hours")
    logger.info(f"Search Query: {query}")

    tweets: list[RawTweet] = []
    start_ts = time.time()
    failed_requests = 0

    try:
        from scraper.twikit_collector import TwikitCollector
        logger.info("Attempting primary collection via Twikit (GraphQL API)...")
        collector = TwikitCollector()
        tweets = asyncio.run(collector.collect(query, since, now, target_count))
        asyncio.run(collector.close())
    except Exception as exc:
        failed_requests += 1
        logger.warning(f"TwikitCollector error: {exc}")

    if not tweets:
        logger.info("Primary Twikit collector returned 0 tweets. Engaging fallback to SeleniumCollector (undetected-chromedriver)...")
        try:
            from scraper.selenium_collector import SeleniumCollector
            collector = SeleniumCollector()
            tweets = asyncio.run(collector.collect(query, since, now, target_count))
            asyncio.run(collector.close())
        except Exception as exc2:
            failed_requests += 1
            logger.error(f"SeleniumCollector also failed: {exc2}")

    elapsed = time.time() - start_ts
    stats = CollectionStats(
        requested=target_count,
        collected=len(tweets),
        duplicate_count=0,
        final_count=len(tweets),
        failed_requests=failed_requests,
        elapsed_seconds=elapsed,
    )
    print(stats.summary())

    if not tweets:
        logger.warning("No tweets collected. Please ensure Twitter credentials or cookies are configured in .env.")
        return None

    out_file = write_raw_tweets(tweets, RAW_DATA_DIR)
    logger.info(f"Saved {len(tweets)} raw tweets to {out_file}")
    return RAW_DATA_DIR


def stage_process(raw_dir: Path | None = None) -> Path:
    """
    Clean and normalize collected data, apply multi-stage deduplication,
    run sentiment analysis, and persist to Parquet format.
    """
    from processing.cleaner import clean_tweet
    from processing.deduplication import Deduplicator
    from processing.storage import read_raw_tweets, write_processed_tweets
    from analysis.sentiment import SentimentAnalyzer

    logger.info("=== PROCESS PHASE: CLEANING, DEDUPLICATION & STORAGE ===")

    raw_dir = raw_dir or RAW_DATA_DIR
    parquet_files = sorted(Path(raw_dir).glob("*.parquet"), key=os.path.getmtime)
    if not parquet_files:
        logger.error(f"No raw Parquet files found in {raw_dir}. Run 'scrape' or 'demo' first.")
        sys.exit(1)

    latest = parquet_files[-1]
    logger.info(f"Reading raw tweets from: {latest.name}")
    raw_tweets = read_raw_tweets(latest)
    logger.info(f"Loaded {len(raw_tweets)} raw tweets for processing")

    dedup = Deduplicator(
        jaccard_threshold=config.processing.dedup_jaccard_threshold,
        minhash_perms=config.processing.dedup_minhash_perms,
        window_size=config.processing.dedup_window_size,
    )
    analyzer = SentimentAnalyzer()
    processed: list[ProcessedTweet] = []

    start_ts = time.time()
    for rt in tqdm(raw_tweets, desc="Processing Tweets"):
        if dedup.is_duplicate(rt.tweet_id, rt.content):
            continue
        pt = clean_tweet(rt)
        pt.sentiment_score = analyzer.analyze(pt.content_clean)
        processed.append(pt)

    elapsed = time.time() - start_ts
    dedup_stats = dedup.get_stats()
    dup_count = dedup_stats["exact_id_dupes"] + dedup_stats["exact_content_dupes"] + dedup_stats["near_dupes"]

    stats = CollectionStats(
        requested=len(raw_tweets),
        collected=len(raw_tweets),
        duplicate_count=dup_count,
        final_count=len(processed),
        failed_requests=0,
        elapsed_seconds=elapsed,
    )
    print(stats.summary())
    logger.info(f"Deduplication breakdown: {dedup_stats}")

    out_file = write_processed_tweets(processed, PROCESSED_DATA_DIR)
    logger.info(f"Saved {len(processed)} processed tweets to {out_file}")
    return PROCESSED_DATA_DIR


def _generate_reports(
    tweets: list[ProcessedTweet],
    signals: list[TradingSignal],
    top_features: list[tuple[str, float]],
    summary: ValidationSummary,
    results: list[ValidationResult],
) -> None:
    """Generate structured markdown, JSON, and CSV reports in REPORTS_DIR."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. signals_detailed.csv
    csv_path = REPORTS_DIR / "signals_detailed.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "tweet_count", "sentiment_mean", "ci_lower_95", "ci_upper_95",
            "volume_anomaly_ratio", "composite_score", "signal", "confidence",
        ])
        for s in signals:
            writer.writerow([
                s.timestamp.isoformat(), s.tweet_volume, f"{s.sentiment_mean:.4f}",
                f"{s.sentiment_ci_lower:.4f}", f"{s.sentiment_ci_upper:.4f}", f"{s.volume_anomaly_ratio:.2f}",
                f"{s.composite_score:.4f}", s.signal, f"{s.confidence:.4f}",
            ])
    logger.info(f"Saved signal CSV report to: {csv_path}")

    # 2. executive_summary.json
    json_path = REPORTS_DIR / "executive_summary.json"
    lang_counts = Counter(t.lang for t in tweets)
    bullish_count = sum(1 for t in tweets if t.sentiment_score > 0.15)
    bearish_count = sum(1 for t in tweets if t.sentiment_score < -0.15)
    neutral_count = len(tweets) - bullish_count - bearish_count

    symbol_counter = Counter()
    for t in tweets:
        for s in t.symbols:
            symbol_counter[s] += 1

    summary_data = {
        "report_generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "total_tweets_analyzed": len(tweets),
        "language_breakdown": dict(lang_counts),
        "sentiment_distribution": {
            "bullish_count": bullish_count,
            "bullish_pct": round(bullish_count / max(len(tweets), 1) * 100, 2),
            "bearish_count": bearish_count,
            "bearish_pct": round(bearish_count / max(len(tweets), 1) * 100, 2),
            "neutral_count": neutral_count,
            "neutral_pct": round(neutral_count / max(len(tweets), 1) * 100, 2),
            "mean_sentiment": round(sum(t.sentiment_score for t in tweets) / max(len(tweets), 1), 4),
        },
        "top_symbols_mentioned": dict(symbol_counter.most_common(10)),
        "top_tfidf_features": [{"term": term, "idf_weight": round(score, 4)} for term, score in top_features[:15]],
        "signal_breakdown": {
            "total_windows": len(signals),
            "buy_signals": sum(1 for s in signals if s.signal == "BUY"),
            "sell_signals": sum(1 for s in signals if s.signal == "SELL"),
            "hold_signals": sum(1 for s in signals if s.signal == "HOLD"),
        },
        "market_validation": {
            "total_signals_evaluated": summary.total_signals,
            "directional_accuracy_pct": round(summary.overall_accuracy * 100, 2),
            "buy_precision_pct": round(summary.buy_precision * 100, 2),
            "sell_precision_pct": round(summary.sell_precision * 100, 2),
            "average_forward_return_pct": round(summary.avg_forward_return * 100, 4),
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved executive summary JSON to: {json_path}")

    # 3. market_intelligence_report.md
    md_path = REPORTS_DIR / "market_intelligence_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Indian Stock Market Tweet Intelligence & Quantitative Signal Report\n\n")
        f.write(f"**Generated At**: `{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`  \n")
        f.write(f"**Total Processed Tweets**: `{len(tweets):,}`  \n")
        f.write(f"**Analysis Window**: `15 Minutes`  \n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Language Distribution**: {', '.join(f'{k.upper()}: {v} ({v/len(tweets)*100:.1f}%)' for k, v in lang_counts.items())}\n")
        f.write(f"- **Mean Market Sentiment**: `{summary_data['sentiment_distribution']['mean_sentiment']:+.4f}`\n")
        f.write(f"- **Sentiment Stance**: 🟢 Bullish: {bullish_count} ({bullish_count/len(tweets)*100:.1f}%) | 🔴 Bearish: {bearish_count} ({bearish_count/len(tweets)*100:.1f}%) | ⚪ Neutral: {neutral_count} ({neutral_count/len(tweets)*100:.1f}%)\n\n")

        f.write("## 2. Top Indian Stock & Index Mentions\n\n")
        f.write("| Rank | Symbol / Index | Mentions | Share (%) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for i, (sym, count) in enumerate(symbol_counter.most_common(10), 1):
            f.write(f"| {i} | `{sym}` | {count:,} | {count/len(tweets)*100:.1f}% |\n")
        f.write("\n")

        f.write("## 3. Top TF-IDF Financial Features\n\n")
        f.write("| Rank | Financial N-Gram | TF-IDF Score |\n")
        f.write("| :--- | :--- | :--- |\n")
        for i, (term, score) in enumerate(top_features[:10], 1):
            f.write(f"| {i} | `{term}` | `{score:.4f}` |\n")
        f.write("\n")

        f.write("## 4. Quantitative Signal Breakdown\n\n")
        f.write(f"- **Total Time Windows (15-min)**: `{len(signals)}`\n")
        f.write(f"- **BUY Signals (95% CI Lower > +0.20)**: `{summary_data['signal_breakdown']['buy_signals']}`\n")
        f.write(f"- **SELL Signals (95% CI Upper < -0.20)**: `{summary_data['signal_breakdown']['sell_signals']}`\n")
        f.write(f"- **HOLD / Inconclusive**: `{summary_data['signal_breakdown']['hold_signals']}`\n\n")

        f.write("## 5. Generated Artifacts & Visualizations\n\n")
        f.write("- 📊 Sentiment Timeline: `output/plots/sentiment_timeline.png`\n")
        f.write("- 🕯️ Sentiment Candlesticks (with 95% CI): `output/plots/sentiment_candlesticks.png`\n")
        f.write("- 🔥 Hourly Activity Heatmap: `output/plots/volume_heatmap.png`\n")
        f.write("- 📈 Composite Signal Dashboard: `output/plots/signal_dashboard.png`\n")
        f.write("- 🏷️ TF-IDF Keywords: `output/plots/top_tfidf_features.png`\n")
        f.write("- 📋 Tabular Signal Data: `output/reports/signals_detailed.csv`\n")

    logger.info(f"Saved comprehensive market report to: {md_path}")


def stage_analyze(proc_dir: Path | None = None) -> None:
    """
    Execute TF-IDF feature extraction, generate quantitative trading signals with 95% CI,
    validate against market price candles, create visualizations, and write reports.
    """
    from processing.storage import read_processed_tweets, write_signals
    from analysis.tfidf_engine import TfidfFeatureEngine
    from analysis.signal_generator import SignalGenerator
    from analysis.market_validator import MarketValidator
    from analysis.visualizer import Visualizer
    from market_data.csv_provider import CSVMarketDataProvider

    logger.info("=== ANALYZE PHASE: SIGNALS, VALIDATION & VISUALIZATION ===")

    proc_dir = proc_dir or PROCESSED_DATA_DIR
    parquet_files = sorted(Path(proc_dir).glob("*.parquet"), key=os.path.getmtime)
    if not parquet_files:
        logger.error(f"No processed Parquet files found in {proc_dir}. Run 'process' first.")
        sys.exit(1)

    latest = parquet_files[-1]
    tweets = read_processed_tweets(latest)
    logger.info(f"Loaded {len(tweets)} processed tweets from: {latest.name}")

    # 1. TF-IDF Domain Feature Extraction
    texts = [t.content_clean for t in tweets]
    tfidf = TfidfFeatureEngine()
    tfidf.fit_transform(texts)
    top_features = tfidf.get_top_features(20)
    top_terms = [f[0] for f in top_features[:10]]
    logger.info(f"Top 10 TF-IDF Features: {top_terms}")

    # 2. Composite Signal Generation with 95% CI
    sig_gen = SignalGenerator()
    signals = sig_gen.generate_signals(tweets, config.analysis.signal_window_minutes)
    buy_count = sum(1 for s in signals if s.signal == "BUY")
    sell_count = sum(1 for s in signals if s.signal == "SELL")
    hold_count = sum(1 for s in signals if s.signal == "HOLD")
    logger.info(f"Generated {len(signals)} Signals (Window={config.analysis.signal_window_minutes}m): BUY={buy_count} | SELL={sell_count} | HOLD={hold_count}")

    # 3. Store Signals to Parquet
    write_signals(signals, SIGNALS_DATA_DIR)

    # 4. Market Price Validation
    market_csv = SAMPLE_DIR / "sample_market_data.csv"
    results: list[ValidationResult] = []
    summary = ValidationSummary()

    if market_csv.exists() and signals:
        provider = CSVMarketDataProvider(market_csv)
        market_data = provider.get_candles(
            "NIFTY50",
            datetime(2026, 8, 14, 9, 15),
            datetime(2026, 8, 14, 15, 30),
        )
        if market_data:
            validator = MarketValidator()
            results, summary = validator.validate(
                signals, market_data, config.analysis.forward_look_minutes,
            )
            print(summary.summary())
        else:
            logger.warning("No market candles matched the signal time range.")
    else:
        logger.info("Market data CSV not found or no signals to validate.")

    # 5. Memory-Efficient Visualizations
    vis = Visualizer()
    vis.plot_sentiment_timeline(tweets, PLOTS_DIR / "sentiment_timeline.png")
    vis.plot_sentiment_candlesticks(signals, PLOTS_DIR / "sentiment_candlesticks.png")
    vis.plot_volume_heatmap(tweets, PLOTS_DIR / "volume_heatmap.png")
    vis.plot_signal_dashboard(signals, PLOTS_DIR / "signal_dashboard.png")
    vis.plot_top_features(top_features, PLOTS_DIR / "top_tfidf_features.png")
    if results and summary.total_signals > 0:
        vis.plot_validation_report(summary, results, PLOTS_DIR / "validation_report.png")

    logger.info(f"All visualization artifacts generated in: {PLOTS_DIR}")

    # 6. Structured Report Generation
    _generate_reports(tweets, signals, top_features, summary, results)
    logger.info(f"All report artifacts generated in: {REPORTS_DIR}")


# ---------------------------------------------------------------------------
# CLI Command Handlers
# ---------------------------------------------------------------------------
def cmd_scrape(target: int | None = None) -> None:
    stage_scrape(target)


def cmd_process() -> None:
    stage_process()


def cmd_analyze() -> None:
    stage_analyze()


def cmd_run(target: int | None = None) -> None:
    raw_dir = stage_scrape(target)
    if raw_dir:
        proc_dir = stage_process(raw_dir)
        stage_analyze(proc_dir)


def cmd_demo() -> None:
    from processing.storage import write_raw_tweets

    logger.info("=== STARTING STANDALONE DEMO MODE (SYNTHETIC DATA) ===")
    logger.info("Simulating realistic 6-hour Indian market intraday trading sessions (250 tweets)...")

    synthetic_tweets = _generate_synthetic_tweets(250)
    write_raw_tweets(synthetic_tweets, RAW_DATA_DIR)

    proc_dir = stage_process(RAW_DATA_DIR)
    stage_analyze(proc_dir)

def cmd_advanced() -> None:
    from analysis.advanced_analytics import run_advanced_analytics
    logger.info("=== EXECUTING ADVANCED QUANTITATIVE ANALYTICS ===")
    results = run_advanced_analytics()
    logger.info("Advanced analytics report generated in output/reports/advanced_quant_insights.md")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indian Stock Market Tweet Intelligence & Trading Signal System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=["login", "scrape", "process", "analyze", "advanced", "run", "demo"],
        help="Pipeline command to execute",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Target number of tweets to collect for scrape/run (default from config: 2000)",
    )

    args = parser.parse_args()

    try:
        if args.command == "login":
            cmd_login()
        elif args.command == "scrape":
            cmd_scrape(args.target)
        elif args.command == "process":
            cmd_process()
        elif args.command == "analyze":
            cmd_analyze()
        elif args.command == "advanced":
            cmd_advanced()
        elif args.command == "run":
            cmd_run(args.target)
        elif args.command == "demo":
            cmd_demo()
    except KeyboardInterrupt:
        logger.info("Interrupted by user -- exiting cleanly.")
        sys.exit(0)
    except Exception:
        logger.exception("Pipeline failed with an unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
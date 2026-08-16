"""
Advanced Quantitative Analytics Module for Indian Stock Market Tweet Intelligence.
Performs:
1. Social Put-Call Ratio (Social PCR) & Strike Distribution
2. Influencer / Tiered Author Alpha & Sentiment Divergence
3. Sentiment Volatility & Market Regime Classification
4. Intraday Trading Session Sentiment Breakdown
5. Stock & Index Co-occurrence Network Analysis
"""
from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PROCESSED_DATA_DIR, PLOTS_DIR, REPORTS_DIR
from models import ProcessedTweet
from processing.storage import read_processed_tweets

logger = logging.getLogger(__name__)


def run_advanced_analytics(file_path: Path | None = None) -> dict:
    """Execute advanced quantitative financial analyses on processed tweets."""
    if file_path:
        target_file = Path(file_path)
    else:
        # Check for full real dataset first, else pick largest file
        real_target = Path(PROCESSED_DATA_DIR) / "tweets_real_2000.parquet"
        if real_target.exists():
            target_file = real_target
        else:
            proc_files = sorted(Path(PROCESSED_DATA_DIR).glob("*.parquet"), key=lambda x: x.stat().st_size)
            if not proc_files:
                raise FileNotFoundError("No processed Parquet files found in data/processed/")
            target_file = proc_files[-1]

    tweets = read_processed_tweets(target_file)
    logger.info(f"Loaded {len(tweets)} tweets for advanced analysis from: {target_file.name}")

    # -----------------------------------------------------------------------
    # 1. Social Put-Call Ratio (Social PCR) & Option Strikes
    # -----------------------------------------------------------------------
    ce_strikes: list[str] = []
    pe_strikes: list[str] = []
    ce_tweets = 0
    pe_tweets = 0

    re_ce = re.compile(r"(\d{5})\s*(?:CE|CALL)", re.IGNORECASE)
    re_pe = re.compile(r"(\d{5})\s*(?:PE|PUT)", re.IGNORECASE)

    for t in tweets:
        ce_found = re_ce.findall(t.content_raw)
        pe_found = re_pe.findall(t.content_raw)
        if ce_found:
            ce_tweets += 1
            ce_strikes.extend(ce_found)
        if pe_found:
            pe_tweets += 1
            pe_strikes.extend(pe_found)

    social_pcr = len(pe_strikes) / max(len(ce_strikes), 1)
    top_ce = Counter(ce_strikes).most_common(5)
    top_pe = Counter(pe_strikes).most_common(5)

    # -----------------------------------------------------------------------
    # 2. Author Influence Tiering (Smart Money vs Retail)
    # -----------------------------------------------------------------------
    tier_1_authors: list[ProcessedTweet] = []  # High followers / engagement (>5000 followers or >50 likes)
    tier_2_authors: list[ProcessedTweet] = []  # Active traders (500-5000 followers)
    tier_3_authors: list[ProcessedTweet] = []  # Retail / New accounts (<500 followers)

    for t in tweets:
        if t.follower_count >= 5000 or t.likes >= 50:
            tier_1_authors.append(t)
        elif t.follower_count >= 500:
            tier_2_authors.append(t)
        else:
            tier_3_authors.append(t)

    t1_mean_sent = sum(t.sentiment_score for t in tier_1_authors) / max(len(tier_1_authors), 1)
    t3_mean_sent = sum(t.sentiment_score for t in tier_3_authors) / max(len(tier_3_authors), 1)
    smart_retail_divergence = t1_mean_sent - t3_mean_sent

    # -----------------------------------------------------------------------
    # 3. Sentiment Volatility & Regime Detection
    # -----------------------------------------------------------------------
    window_size = 50
    sentiment_scores = [t.sentiment_score for t in sorted(tweets, key=lambda x: x.timestamp)]
    rolling_vols = []
    for i in range(len(sentiment_scores) - window_size + 1):
        chunk = sentiment_scores[i : i + window_size]
        rolling_vols.append(float(np.std(chunk)))

    mean_sent_vol = float(np.mean(rolling_vols)) if rolling_vols else 0.0
    regime = "High Volatility / Directional Uncertainty" if mean_sent_vol > 0.40 else "Consolidation / Stable Consensus"

    # -----------------------------------------------------------------------
    # 4. Intraday Market Session Breakdown (IST UTC+5:30)
    # -----------------------------------------------------------------------
    session_counts: Counter[str] = Counter()
    session_sentiments: defaultdict[str, list[float]] = defaultdict(list)

    for t in tweets:
        # Convert UTC to IST (+5.5h)
        hour_ist = (t.timestamp.hour + 5 + (t.timestamp.minute + 30) // 60) % 24
        if 8 <= hour_ist < 9.25:
            sess = "Pre-Market (08:00-09:15)"
        elif 9.25 <= hour_ist < 10.5:
            sess = "Opening Bell (09:15-10:30)"
        elif 10.5 <= hour_ist < 13.5:
            sess = "Mid-Day Consolidation (10:30-13:30)"
        elif 13.5 <= hour_ist < 15.5:
            sess = "Closing Rush (13:30-15:30)"
        else:
            sess = "Post-Market / Evening"

        session_counts[sess] += 1
        session_sentiments[sess].append(t.sentiment_score)

    session_stats = {
        s: {
            "tweet_count": session_counts[s],
            "mean_sentiment": round(float(np.mean(session_sentiments[s])), 4) if session_sentiments[s] else 0.0,
        }
        for s in [
            "Pre-Market (08:00-09:15)",
            "Opening Bell (09:15-10:30)",
            "Mid-Day Consolidation (10:30-13:30)",
            "Closing Rush (13:30-15:30)",
            "Post-Market / Evening",
        ]
    }

    # -----------------------------------------------------------------------
    # 5. Entity Co-Occurrence Network
    # -----------------------------------------------------------------------
    co_occurrences: Counter[tuple[str, str]] = Counter()
    for t in tweets:
        syms = sorted(set(t.symbols))
        if len(syms) >= 2:
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    co_occurrences[(syms[i], syms[j])] += 1

    top_co_occurrences = [
        {"pair": f"{k[0]} - {k[1]}", "count": v}
        for k, v in co_occurrences.most_common(8)
    ]

    # -----------------------------------------------------------------------
    # 6. Generate Multi-Panel Visualization Dashboard
    # -----------------------------------------------------------------------
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / "advanced_analytics_dashboard.png"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panel 1: Option Strike Distribution & Social PCR
    ax1 = axes[0, 0]
    strikes_labels = [f"CE {s[0]}" for s in top_ce[:4]] + [f"PE {s[0]}" for s in top_pe[:4]]
    strikes_counts = [s[1] for s in top_ce[:4]] + [s[1] for s in top_pe[:4]]
    colors1 = ["#2ecc71"] * min(4, len(top_ce)) + ["#e74c3c"] * min(4, len(top_pe))

    if strikes_labels:
        bars = ax1.barh(strikes_labels, strikes_counts, color=colors1, edgecolor="#2c3e50")
        ax1.set_title(f"Options Strike Mentions (Social PCR: {social_pcr:.3f})", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Mention Frequency")
        ax1.grid(axis="x", linestyle="--", alpha=0.6)
        for bar in bars:
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{int(bar.get_width())}", va="center", fontsize=9)
    else:
        ax1.text(0.5, 0.5, "No specific strikes detected", ha="center", va="center")

    # Panel 2: Author Tier Sentiment Divergence
    ax2 = axes[0, 1]
    tiers = ["Tier 1: High Influence\n(>5K Followers)", "Tier 2: Active Traders\n(500-5K Followers)", "Tier 3: Retail / Broad\n(<500 Followers)"]
    tier_sents = [t1_mean_sent, sum(t.sentiment_score for t in tier_2_authors)/max(len(tier_2_authors),1), t3_mean_sent]
    tier_colors = ["#3498db", "#9b59b6", "#e67e22"]

    bars2 = ax2.bar(tiers, tier_sents, color=tier_colors, edgecolor="#2c3e50", width=0.55)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title(f"Sentiment by Influencer Tier (Divergence: {smart_retail_divergence:+.3f})", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean Sentiment Score [-1, +1]")
    ax2.set_ylim(-0.25, 0.35)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars2:
        val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, val + (0.015 if val >= 0 else -0.03), f"{val:+.3f}", ha="center", fontweight="bold", fontsize=10)

    # Panel 3: Intraday Session Sentiment
    ax3 = axes[1, 0]
    sess_names = [s.split(" (")[0] for s in session_stats.keys()]
    sess_means = [session_stats[s]["mean_sentiment"] for s in session_stats.keys()]
    sess_vols = [session_stats[s]["tweet_count"] for s in session_stats.keys()]

    ax3_twin = ax3.twinx()
    ax3.bar(sess_names, sess_vols, color="#bdc3c7", alpha=0.5, label="Tweet Volume")
    ax3_twin.plot(sess_names, sess_means, color="#e74c3c", marker="o", linewidth=2.5, label="Mean Sentiment")
    ax3.set_title("Intraday Market Session Dynamics (IST)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Tweet Volume", color="#7f8c8d")
    ax3_twin.set_ylabel("Mean Sentiment", color="#e74c3c")
    ax3.tick_params(axis="x", rotation=15)
    ax3.grid(axis="x", linestyle="--", alpha=0.4)

    # Panel 4: Entity Co-Occurrence Network
    ax4 = axes[1, 1]
    if top_co_occurrences:
        pair_names = [p["pair"] for p in top_co_occurrences[:6]]
        pair_counts = [p["count"] for p in top_co_occurrences[:6]]
        ax4.barh(pair_names, pair_counts, color="#1abc9c", edgecolor="#2c3e50")
        ax4.set_title("Top Stock & Index Co-Occurrences", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Co-Mention Count")
        ax4.grid(axis="x", linestyle="--", alpha=0.6)
    else:
        ax4.text(0.5, 0.5, "Insufficient co-occurrences", ha="center", va="center")

    plt.suptitle("Dalal Street Advanced Quantitative Sentiment Intelligence", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved advanced dashboard visualization to: {plot_path}")

    # -----------------------------------------------------------------------
    # 7. Write Comprehensive Markdown Report
    # -----------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "advanced_quant_insights.md"

    results = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "total_tweets_analyzed": len(tweets),
        "social_put_call_ratio": {
            "pcr_ratio": round(social_pcr, 4),
            "ce_mentions": len(ce_strikes),
            "pe_mentions": len(pe_strikes),
            "top_ce_strikes": top_ce,
            "top_pe_strikes": top_pe,
        },
        "author_tier_divergence": {
            "tier_1_high_influence": {"count": len(tier_1_authors), "mean_sentiment": round(t1_mean_sent, 4)},
            "tier_2_active_traders": {"count": len(tier_2_authors), "mean_sentiment": round(sum(t.sentiment_score for t in tier_2_authors)/max(len(tier_2_authors),1), 4)},
            "tier_3_retail": {"count": len(tier_3_authors), "mean_sentiment": round(t3_mean_sent, 4)},
            "smart_retail_spread": round(smart_retail_divergence, 4),
        },
        "sentiment_volatility": {
            "rolling_std_mean": round(mean_sent_vol, 4),
            "market_regime": regime,
        },
        "session_breakdown": session_stats,
        "top_co_occurrences": top_co_occurrences,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Advanced Quantitative Tweet Intelligence & Market Insights\n\n")
        f.write(f"**Generated At**: `{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`  \n")
        f.write(f"**Dataset Size**: `{len(tweets):,}` Unique Real Tweets  \n\n")

        f.write("## 1. Social Put-Call Ratio (Social PCR) & Strike Sentiment\n\n")
        f.write(f"- **Social PCR (PE / CE Ratio)**: `{social_pcr:.3f}` *(Values < 0.7 indicate strong Call-buying / Bullish bias)*\n")
        f.write(f"- **Total CE (Call) Mentions**: `{len(ce_strikes)}` across {ce_tweets} tweets\n")
        f.write(f"- **Total PE (Put) Mentions**: `{len(pe_strikes)}` across {pe_tweets} tweets\n")
        f.write(f"- **Key Call Resistance Strikes**: {', '.join(f'`{s[0]}` ({s[1]}x)' for s in top_ce[:4])}\n")
        f.write(f"- **Key Put Support Strikes**: {', '.join(f'`{s[0]}` ({s[1]}x)' for s in top_pe[:4])}\n\n")

        f.write("## 2. Influencer Tiering & Smart vs. Retail Sentiment Spread\n\n")
        f.write("| Tier | Classification | Sample Count | Mean Sentiment |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Tier 1** | High Influence (>5K Followers / >50 Likes) | {len(tier_1_authors):,} | `{t1_mean_sent:+.4f}` |\n")
        f.write(f"| **Tier 2** | Active Market Participants (500-5K Followers) | {len(tier_2_authors):,} | `{results['author_tier_divergence']['tier_2_active_traders']['mean_sentiment']:+.4f}` |\n")
        f.write(f"| **Tier 3** | Retail / Broad Community (<500 Followers) | {len(tier_3_authors):,} | `{t3_mean_sent:+.4f}` |\n\n")
        f.write(f"> **Smart-Money vs Retail Sentiment Spread**: `{smart_retail_divergence:+.4f}`\n\n")

        f.write("## 3. Market Regime & Sentiment Volatility\n\n")
        f.write(f"- **Rolling Sentiment Standard Deviation**: `{mean_sent_vol:.4f}`\n")
        f.write(f"- **Detected Market Regime**: `{regime}`\n\n")

        f.write("## 4. Intraday Market Session Dynamics (IST)\n\n")
        f.write("| Market Session (IST) | Tweet Volume | Mean Sentiment |\n")
        f.write("| :--- | :--- | :--- |\n")
        for sess, data in session_stats.items():
            f.write(f"| `{sess}` | {data['tweet_count']:,} | `{data['mean_sentiment']:+.4f}` |\n")
        f.write("\n")

        f.write("## 5. Stock & Index Co-Occurrence Network\n\n")
        f.write("| Stock / Index Pair | Co-Mention Frequency |\n")
        f.write("| :--- | :--- |\n")
        for item in top_co_occurrences[:6]:
            f.write(f"| `{item['pair']}` | {item['count']:,} |\n")
        f.write("\n")

        f.write("## 6. Generated Visual Artifacts\n\n")
        f.write(f"- 📊 **Multi-Panel Quantitative Dashboard**: `output/plots/advanced_analytics_dashboard.png`\n")

    # Save JSON
    json_path = REPORTS_DIR / "advanced_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved advanced report to: {report_path}")
    logger.info(f"Saved advanced JSON to: {json_path}")
    return results


if __name__ == "__main__":
    run_advanced_analytics()

"""
Memory-efficient visualization utilities for tweet sentiment and trading signals.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt
import numpy as np

from models import ProcessedTweet, TradingSignal, ValidationResult, ValidationSummary
from config import config

logger = logging.getLogger(__name__)


def largest_triangle_three_buckets(
    data: List[Tuple[datetime, float]],
    target_points: int,
) -> List[Tuple[datetime, float]]:
    """
    Largest-Triangle-Three-Buckets (LTTB) downsampling algorithm.
    Reduces time-series points while preserving visual extrema and trends.
    """
    if len(data) <= target_points or target_points < 3:
        return data

    sampled = [data[0]]
    every = (len(data) - 2) / (target_points - 2)
    a = 0

    for i in range(target_points - 2):
        avg_x = 0.0
        avg_y = 0.0
        avg_range_start = int((i + 1) * every) + 1
        avg_range_end = min(int((i + 2) * every) + 1, len(data))
        avg_range_length = max(avg_range_end - avg_range_start, 1)

        for idx in range(avg_range_start, avg_range_end):
            avg_x += data[idx][0].timestamp()
            avg_y += data[idx][1]

        avg_x /= avg_range_length
        avg_y /= avg_range_length

        range_offs = int(i * every) + 1
        range_to = min(int((i + 1) * every) + 1, len(data))

        point_a_x = data[a][0].timestamp()
        point_a_y = data[a][1]

        max_area = -1.0
        max_area_point = None
        next_a = range_offs

        for idx in range(range_offs, range_to):
            area = abs(
                (point_a_x - avg_x) * (data[idx][1] - point_a_y)
                - (point_a_x - data[idx][0].timestamp()) * (avg_y - point_a_y)
            ) * 0.5

            if area > max_area:
                max_area = area
                max_area_point = data[idx]
                next_a = idx

        if max_area_point is not None:
            sampled.append(max_area_point)
            a = next_a
        elif range_offs < len(data):
            sampled.append(data[range_offs])
            a = range_offs

    sampled.append(data[-1])
    return sampled


class Visualizer:
    """Generates publication-quality, memory-efficient visual charts."""

    def __init__(self) -> None:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            plt.style.use('default')

    def plot_sentiment_timeline(self, tweets: list[ProcessedTweet], save_path: Path | str) -> None:
        """Plot sentiment trajectory over time with LTTB downsampling for large datasets."""
        try:
            if not tweets:
                return
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            sorted_tweets = sorted(tweets, key=lambda x: x.timestamp)
            data = [(t.timestamp, t.sentiment_score) for t in sorted_tweets]
            if len(data) > config.analysis.lttb_target_points:
                data = largest_triangle_three_buckets(data, config.analysis.lttb_target_points)

            x = [d[0] for d in data]
            y = [d[1] for d in data]

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(x, y, color='#1f77b4', alpha=0.7, linewidth=1.2, label='Tweet Sentiment')
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.set_title('Indian Stock Market Tweet Sentiment Timeline', fontsize=14, fontweight='bold')
            ax.set_xlabel('Timestamp (UTC)', fontsize=11)
            ax.set_ylabel('Sentiment Score (-1.0 to +1.0)', fontsize=11)
            ax.set_ylim(-1.05, 1.05)
            ax.legend(loc='upper right')
            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved sentiment timeline to {save_path}")
        except Exception as e:
            logger.error(f"Error plotting sentiment timeline: {e}")

    def plot_sentiment_candlesticks(self, signals: list[TradingSignal], save_path: Path | str) -> None:
        """Plot aggregated window sentiment mean with 95% confidence intervals."""
        try:
            if not signals:
                return
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            x = [s.timestamp for s in signals]
            mu = [s.sentiment_mean for s in signals]
            ci_l = [s.sentiment_ci_lower for s in signals]
            ci_u = [s.sentiment_ci_upper for s in signals]

            yerr_lower = [max(0.0, m - l) for m, l in zip(mu, ci_l)]
            yerr_upper = [max(0.0, u - m) for u, m in zip(ci_u, mu)]

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.errorbar(
                x, mu, yerr=[yerr_lower, yerr_upper],
                fmt='o', color='#6f42c1', ecolor='#6c757d',
                elinewidth=1.5, capsize=4, label='Mean Sentiment (95% CI)',
            )
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.axhline(config.analysis.signal_buy_threshold, color='#28a745', linestyle=':', label='BUY Threshold (+0.2)')
            ax.axhline(config.analysis.signal_sell_threshold, color='#dc3545', linestyle=':', label='SELL Threshold (-0.2)')
            ax.set_title('Windowed Sentiment with 95% Confidence Intervals', fontsize=14, fontweight='bold')
            ax.set_xlabel('Window Timestamp', fontsize=11)
            ax.set_ylabel('Sentiment Score', fontsize=11)
            ax.set_ylim(-1.1, 1.1)
            ax.legend(loc='upper right')
            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved sentiment candlesticks to {save_path}")
        except Exception as e:
            logger.error(f"Error plotting sentiment candlesticks: {e}")

    def plot_volume_heatmap(self, tweets: list[ProcessedTweet], save_path: Path | str) -> None:
        """Plot tweet volume distribution across time intervals."""
        try:
            if not tweets:
                return
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            hours = [t.timestamp.replace(minute=(t.timestamp.minute // 15) * 15, second=0, microsecond=0) for t in tweets]
            from collections import Counter
            counts = Counter(hours)
            if not counts:
                return

            sorted_hours = sorted(counts.keys())
            x = sorted_hours
            y = [counts[h] for h in sorted_hours]

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(x, y, width=0.008, color='#fd7e14', alpha=0.85, edgecolor='#d96509')
            ax.set_title('Intraday Tweet Activity Volume Distribution (15m Bins)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Window', fontsize=11)
            ax.set_ylabel('Tweet Volume', fontsize=11)
            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved volume heatmap to {save_path}")
        except Exception as e:
            logger.error(f"Error plotting volume heatmap: {e}")

    def plot_signal_dashboard(self, signals: list[TradingSignal], save_path: Path | str) -> None:
        """Plot comprehensive dashboard of signals, composite score, and threshold bands."""
        try:
            if not signals:
                return
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            x = [s.timestamp for s in signals]
            comp = [s.composite_score for s in signals]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(x, comp, color='#343a40', linewidth=1.5, label='Composite Signal Score')

            buy_x = [s.timestamp for s in signals if s.signal == "BUY"]
            buy_y = [s.composite_score for s in signals if s.signal == "BUY"]
            sell_x = [s.timestamp for s in signals if s.signal == "SELL"]
            sell_y = [s.composite_score for s in signals if s.signal == "SELL"]

            if buy_x:
                ax.scatter(buy_x, buy_y, color='#28a745', marker='^', s=120, zorder=5, label='BUY Signal')
            if sell_x:
                ax.scatter(sell_x, sell_y, color='#dc3545', marker='v', s=120, zorder=5, label='SELL Signal')

            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.axhline(config.analysis.signal_buy_threshold, color='#28a745', linestyle=':', alpha=0.7)
            ax.axhline(config.analysis.signal_sell_threshold, color='#dc3545', linestyle=':', alpha=0.7)

            ax.set_title('Composite Trading Signal Dashboard (BUY / SELL / HOLD Zones)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Timestamp', fontsize=11)
            ax.set_ylabel('Composite Score', fontsize=11)
            ax.legend(loc='upper right')
            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved signal dashboard to {save_path}")
        except Exception as e:
            logger.error(f"Error plotting signal dashboard: {e}")

    def plot_validation_report(
        self,
        summary: ValidationSummary,
        results: list[ValidationResult],
        save_path: Path | str,
    ) -> None:
        """Plot market validation scorecard with precision metrics and return distribution."""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Precision & Accuracy Bar Chart
            labels = ['BUY Precision', 'SELL Precision', 'Overall Accuracy', 'Signal Coverage']
            vals = [summary.buy_precision, summary.sell_precision, summary.overall_accuracy, summary.signal_coverage]
            colors = ['#28a745', '#dc3545', '#007bff', '#6f42c1']

            bars = ax1.bar(labels, vals, color=colors, alpha=0.85, width=0.55)
            ax1.set_ylim(0, 1.05)
            ax1.set_title('Market Signal Precision & Accuracy', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Ratio (0.0 to 1.0)', fontsize=11)

            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{height:.1%}", ha='center', va='bottom', fontsize=10)

            # Return Distribution Histogram
            buy_rets = [r.forward_return_pct for r in results if r.signal_direction == "BUY"]
            sell_rets = [r.forward_return_pct for r in results if r.signal_direction == "SELL"]

            if buy_rets or sell_rets:
                if buy_rets:
                    ax2.hist(buy_rets, bins=10, alpha=0.6, color='#28a745', label=f'BUY Returns (avg {summary.avg_buy_return:+.2f}%)')
                if sell_rets:
                    ax2.hist(sell_rets, bins=10, alpha=0.6, color='#dc3545', label=f'SELL Returns (avg {summary.avg_sell_return:+.2f}%)')
                ax2.axvline(0, color='black', linestyle='--', linewidth=0.8)
                ax2.set_title('Forward Return Distribution (15m Horizon)', fontsize=13, fontweight='bold')
                ax2.set_xlabel('Return (%)', fontsize=11)
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'No actionable signals to backtest', ha='center', va='center', transform=ax2.transAxes, fontsize=12, color='gray')
                ax2.set_title('Forward Return Distribution', fontsize=13, fontweight='bold')

            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved validation report to {save_path}")
        except Exception as e:
            logger.error(f"Error plotting validation report: {e}")

    def plot_top_features(self, features: list[tuple[str, float]], save_path: Path | str) -> None:
        """Plot top TF-IDF financial keywords by weight."""
        try:
            if not features:
                return
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            features_sorted = sorted(features, key=lambda x: x[1])[-15:]  # Top 15
            words = [f[0] for f in features_sorted]
            scores = [f[1] for f in features_sorted]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(words, scores, color='#17a2b8', alpha=0.85)
            ax.set_title('Top Domain TF-IDF Keywords & Sentiment Features', fontsize=13, fontweight='bold')
            ax.set_xlabel('IDF Weight / Importance Score', fontsize=11)
            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved top features to {save_path}")
        except Exception as e:
            logger.error(f"Error plotting top features: {e}")

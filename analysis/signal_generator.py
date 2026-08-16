"""
Trading signal generator based on tweet sentiment and volume with 95% Confidence Intervals.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
import numpy as np

from models import ProcessedTweet, TradingSignal
from config import config

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates composite trading signals with 95% confidence intervals from processed tweets."""

    def __init__(self) -> None:
        self.half_life = config.analysis.signal_half_life_minutes
        self.window_minutes = config.analysis.signal_window_minutes
        self.buy_threshold = config.analysis.signal_buy_threshold
        self.sell_threshold = config.analysis.signal_sell_threshold
        self.min_volume_ratio = config.analysis.signal_min_volume_ratio

    def generate_signals(
        self,
        tweets: list[ProcessedTweet],
        window_minutes: int | None = None,
    ) -> list[TradingSignal]:
        """
        Generate composite trading signals from tweets grouped into time windows.

        Calculates:
        1. Engagement weighting: w_i = ln(1 + likes + 2*retweets + 0.5*replies) * ln(1 + followers)
        2. Exponential time decay: omega_i(t) = w_i * exp(-lambda * (w_end - t_i))
        3. Weighted sentiment mean: mu_t = sum(omega_i * S_i) / sum(omega_i)
        4. 95% CI via Weighted SEM: N_eff = (sum omega)^2 / sum(omega^2), s_w^2 = weighted variance, CI = mu +- 1.96 * s_w / sqrt(N_eff)
        5. Bull-Bear Ratio: (N_bull - N_bear) / (N_bull + N_bear + eps)
        6. Sentiment Velocity: mu_t - mu_{t-1}
        7. Volume Anomaly Ratio: window_volume / mean_volume
        8. Signal: BUY if CI_lower > buy_threshold AND VAR > min_volume_ratio,
                   SELL if CI_upper < sell_threshold AND VAR > min_volume_ratio,
                   else HOLD
        """
        if not tweets:
            return []

        w_minutes = window_minutes or self.window_minutes
        sorted_tweets = sorted(tweets, key=lambda x: x.timestamp)
        start_time = sorted_tweets[0].timestamp
        end_time = sorted_tweets[-1].timestamp

        # O(1) mathematical bucket grouping
        windows: dict[datetime, list[ProcessedTweet]] = {}
        curr = start_time
        while curr <= end_time:
            windows[curr] = []
            curr += timedelta(minutes=w_minutes)

        w_seconds = w_minutes * 60
        for t in sorted_tweets:
            offset_sec = max(0.0, (t.timestamp - start_time).total_seconds())
            idx = int(offset_sec // w_seconds)
            w_key = start_time + timedelta(minutes=idx * w_minutes)
            if w_key in windows:
                windows[w_key].append(t)
            else:
                windows[w_key] = [t]

        signals: list[TradingSignal] = []
        lambd = np.log(2) / max(self.half_life, 1e-6)

        window_volumes = [len(w) for w in windows.values()]
        mean_volume = float(np.mean(window_volumes)) if window_volumes else 0.0

        prev_mu = 0.0

        for w_start, w_tweets in sorted(windows.items()):
            w_end = w_start + timedelta(minutes=w_minutes)
            if not w_tweets:
                continue

            weights = []
            sentiments = []
            n_bull = 0
            n_bear = 0

            for t in w_tweets:
                # Engagement weighting
                engagement_factor = np.log1p(t.likes + 2.0 * t.retweets + 0.5 * t.replies)
                follower_factor = np.log1p(max(t.follower_count, 0))
                w_i = max(engagement_factor * follower_factor, 0.01)

                # Exponential time decay towards window end
                time_diff_minutes = max((w_end - t.timestamp).total_seconds() / 60.0, 0.0)
                omega_i = w_i * np.exp(-lambd * time_diff_minutes)

                weights.append(omega_i)
                sentiments.append(t.sentiment_score)

                if t.sentiment_score > 0.1:
                    n_bull += 1
                elif t.sentiment_score < -0.1:
                    n_bear += 1

            weights_arr = np.array(weights, dtype=np.float64)
            sentiments_arr = np.array(sentiments, dtype=np.float64)

            sum_w = float(np.sum(weights_arr))
            if sum_w <= 0:
                continue

            # Weighted sentiment mean
            mu_t = float(np.sum(weights_arr * sentiments_arr) / sum_w)

            # Effective sample size (Kish's formula)
            sum_w2 = float(np.sum(weights_arr ** 2))
            N_eff = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else 1.0

            # Weighted sample variance & standard error
            if len(w_tweets) > 1 and N_eff > 1.0:
                variance = float(np.average((sentiments_arr - mu_t) ** 2, weights=weights_arr))
                s_w = np.sqrt(variance)
                sem = s_w / np.sqrt(N_eff)
            else:
                s_w = 0.0
                sem = 0.0

            ci_half_width = 1.96 * sem
            ci_lower = float(mu_t - ci_half_width)
            ci_upper = float(mu_t + ci_half_width)

            # Bull-Bear Ratio & Velocity
            bbr = float((n_bull - n_bear) / (n_bull + n_bear + 1e-6))
            vel = float(mu_t - prev_mu)

            # Volume Anomaly Ratio
            var = float(len(w_tweets) / mean_volume) if mean_volume > 0 else 1.0

            # Composite Score
            comp_score = float(mu_t + 0.5 * bbr + 0.2 * vel)

            # Signal Decision Logic
            sig_direction = "HOLD"
            if ci_lower > self.buy_threshold and var >= self.min_volume_ratio:
                sig_direction = "BUY"
            elif ci_upper < self.sell_threshold and var >= self.min_volume_ratio:
                sig_direction = "SELL"

            conf = float(max(0.0, min(1.0, 1.0 - ci_half_width / 2.0)))

            signals.append(TradingSignal(
                timestamp=w_end,
                window_start=w_start,
                window_end=w_end,
                sentiment_mean=mu_t,
                sentiment_ci_lower=ci_lower,
                sentiment_ci_upper=ci_upper,
                tweet_volume=len(w_tweets),
                bull_bear_ratio=bbr,
                sentiment_velocity=vel,
                volume_anomaly_ratio=var,
                composite_score=comp_score,
                signal=sig_direction,
                confidence=conf,
            ))

            prev_mu = mu_t

        return signals

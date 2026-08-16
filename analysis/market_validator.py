"""
Market validator to check trading signals against actual market OHLCV candle data.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from models import TradingSignal, MarketCandle, ValidationResult, ValidationSummary

logger = logging.getLogger(__name__)


class MarketValidator:
    """Validates quantitative trading signals against forward market price movements."""

    def validate(
        self,
        signals: list[TradingSignal],
        market_data: list[MarketCandle],
        forward_minutes: int = 15,
    ) -> tuple[list[ValidationResult], ValidationSummary]:
        """
        Validate signals against forward market returns.

        - BUY is correct if forward price return > 0
        - SELL is correct if forward price return < 0
        """
        results: list[ValidationResult] = []

        if not market_data:
            logger.warning("No market data provided for validation.")
            return [], ValidationSummary()

        # Build lookup table with naive timestamps for consistent comparison
        market_dict: dict[datetime, MarketCandle] = {}
        for candle in market_data:
            ts = candle.timestamp.replace(tzinfo=None) if candle.timestamp.tzinfo is not None else candle.timestamp
            market_dict[ts] = candle

        sorted_times = sorted(market_dict.keys())
        if not sorted_times:
            return [], ValidationSummary()

        def _get_closest_candle(target_time: datetime, max_delta_seconds: int = 900) -> MarketCandle | None:
            t = target_time.replace(tzinfo=None) if target_time.tzinfo is not None else target_time
            closest_ts = min(sorted_times, key=lambda d: abs((d - t).total_seconds()))
            if abs((closest_ts - t).total_seconds()) > max_delta_seconds:
                return None
            return market_dict[closest_ts]

        buy_sig_count = 0
        sell_sig_count = 0
        buy_corr = 0
        sell_corr = 0
        buy_ret_sum = 0.0
        sell_ret_sum = 0.0

        for sig in signals:
            if sig.signal == "HOLD":
                continue

            entry_candle = _get_closest_candle(sig.timestamp)
            exit_time = sig.timestamp + timedelta(minutes=forward_minutes)
            exit_candle = _get_closest_candle(exit_time)

            if not entry_candle or not exit_candle or entry_candle.close == 0:
                continue

            entry_price = entry_candle.close
            exit_price = exit_candle.close
            fwd_ret = (exit_price - entry_price) / entry_price

            is_correct = False
            if sig.signal == "BUY":
                buy_sig_count += 1
                if fwd_ret > 0:
                    is_correct = True
                    buy_corr += 1
                buy_ret_sum += fwd_ret
            elif sig.signal == "SELL":
                sell_sig_count += 1
                if fwd_ret < 0:
                    is_correct = True
                    sell_corr += 1
                sell_ret_sum += (-fwd_ret)  # Positive return from successful short

            results.append(ValidationResult(
                signal_timestamp=sig.timestamp,
                signal_direction=sig.signal,
                forward_return_pct=fwd_ret * 100.0,
                is_correct=is_correct,
            ))

        total_actionable = buy_sig_count + sell_sig_count
        buy_precision = float(buy_corr / buy_sig_count) if buy_sig_count > 0 else 0.0
        sell_precision = float(sell_corr / sell_sig_count) if sell_sig_count > 0 else 0.0
        overall_accuracy = float((buy_corr + sell_corr) / total_actionable) if total_actionable > 0 else 0.0
        coverage = float(total_actionable / len(signals)) if signals else 0.0

        avg_fwd = float((buy_ret_sum + sell_ret_sum) / total_actionable) if total_actionable > 0 else 0.0
        avg_buy = float(buy_ret_sum / buy_sig_count) if buy_sig_count > 0 else 0.0
        avg_sell = float(sell_ret_sum / sell_sig_count) if sell_sig_count > 0 else 0.0

        summary = ValidationSummary(
            total_signals=total_actionable,
            buy_signals=buy_sig_count,
            sell_signals=sell_sig_count,
            buy_correct=buy_corr,
            sell_correct=sell_corr,
            buy_precision=buy_precision,
            sell_precision=sell_precision,
            overall_accuracy=overall_accuracy,
            signal_coverage=coverage,
            avg_forward_return=avg_fwd,
            avg_buy_return=avg_buy,
            avg_sell_return=avg_sell,
        )

        return results, summary

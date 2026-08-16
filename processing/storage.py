"""
Parquet read/write utilities using PyArrow.
"""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from typing import List

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

from models import RawTweet, ProcessedTweet, TradingSignal
from config import config, RAW_DATA_DIR, PROCESSED_DATA_DIR, SIGNALS_DATA_DIR

logger = logging.getLogger(__name__)

# Schema definitions
if pa:
    TWEET_SCHEMA = pa.schema([
        ('tweet_id', pa.string()),
        ('timestamp', pa.timestamp('us', tz='UTC')),
        ('username', pa.string()),
        ('display_name', pa.string()),
        ('content_raw', pa.string()),
        ('content_clean', pa.string()),
        ('lang', pa.dictionary(pa.int8(), pa.string())),
        ('symbols', pa.list_(pa.string())),
        ('hashtags', pa.list_(pa.string())),
        ('mentions', pa.list_(pa.string())),
        ('likes', pa.int32()),
        ('retweets', pa.int32()),
        ('replies', pa.int32()),
        ('views', pa.int32()),
        ('follower_count', pa.int32()),
        ('sentiment_score', pa.float32()),
        ('is_duplicate', pa.bool_())
    ])


def _generate_filename(prefix: str) -> str:
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp_str}.parquet"


def _resolve_target_path(path: Path | str, prefix: str) -> Path:
    target = Path(path)
    if target.suffix == ".parquet":
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    target.mkdir(parents=True, exist_ok=True)
    return target / _generate_filename(prefix)


def write_raw_tweets(tweets: List[RawTweet], path: Path | str = RAW_DATA_DIR) -> Path | None:
    if not pa:
        logger.error("PyArrow not installed, cannot write raw tweets.")
        return None
    if not tweets:
        return None
    try:
        file_path = _resolve_target_path(path, "raw_tweets")
        data = {
            'tweet_id': [str(t.tweet_id) for t in tweets],
            'timestamp': [t.timestamp for t in tweets],
            'username': [t.username for t in tweets],
            'display_name': [t.display_name for t in tweets],
            'content': [t.content for t in tweets],
            'likes': [int(t.likes) for t in tweets],
            'retweets': [int(t.retweets) for t in tweets],
            'replies': [int(t.replies) for t in tweets],
            'views': [int(t.views) if t.views is not None else 0 for t in tweets],
            'hashtags': [t.hashtags for t in tweets],
            'mentions': [t.mentions for t in tweets],
            'follower_count': [int(t.follower_count) for t in tweets],
            'url': [t.url for t in tweets]
        }
        table = pa.Table.from_pydict(data)
        pq.write_table(table, file_path, compression='ZSTD')
        logger.info(f"Wrote {len(tweets)} raw tweets to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to write raw tweets: {e}")
        return None


def read_raw_tweets(path: Path | str) -> List[RawTweet]:
    if not pa:
        logger.error("PyArrow not installed, cannot read raw tweets.")
        return []
    try:
        path = Path(path)
        if not path.exists():
            return []

        table = pq.read_table(path)
        df = table.to_pandas()
        tweets = []
        for _, row in df.iterrows():
            tweets.append(RawTweet(
                tweet_id=str(row.get('tweet_id', '')),
                username=row.get('username', ''),
                display_name=row.get('display_name', ''),
                timestamp=row.get('timestamp', None),
                content=row.get('content', ''),
                likes=int(row.get('likes', 0)),
                retweets=int(row.get('retweets', 0)),
                replies=int(row.get('replies', 0)),
                views=int(row.get('views', 0)) if row.get('views') is not None else None,
                hashtags=list(row.get('hashtags', [])) if row.get('hashtags') is not None else [],
                mentions=list(row.get('mentions', [])) if row.get('mentions') is not None else [],
                follower_count=int(row.get('follower_count', 0)),
                url=str(row.get('url', ''))
            ))
        return tweets
    except Exception as e:
        logger.error(f"Failed to read raw tweets from {path}: {e}")
        return []


def write_processed_tweets(tweets: List[ProcessedTweet], path: Path | str = PROCESSED_DATA_DIR) -> Path | None:
    if not pa:
        logger.error("PyArrow not installed, cannot write processed tweets.")
        return None
    if not tweets:
        return None
    try:
        file_path = _resolve_target_path(path, "tweets")
        data = {
            'tweet_id': [str(t.tweet_id) for t in tweets],
            'timestamp': [t.timestamp for t in tweets],
            'username': [t.username for t in tweets],
            'display_name': [t.display_name for t in tweets],
            'content_raw': [t.content_raw for t in tweets],
            'content_clean': [t.content_clean for t in tweets],
            'lang': [t.lang for t in tweets],
            'symbols': [t.symbols for t in tweets],
            'hashtags': [t.hashtags for t in tweets],
            'mentions': [t.mentions for t in tweets],
            'likes': [int(t.likes) for t in tweets],
            'retweets': [int(t.retweets) for t in tweets],
            'replies': [int(t.replies) for t in tweets],
            'views': [int(t.views) if t.views is not None else 0 for t in tweets],
            'follower_count': [int(t.follower_count) for t in tweets],
            'sentiment_score': [float(t.sentiment_score) for t in tweets],
            'is_duplicate': [bool(t.is_duplicate) for t in tweets]
        }
        table = pa.Table.from_pydict(data)
        pq.write_table(table, file_path, compression='ZSTD')
        logger.info(f"Wrote {len(tweets)} processed tweets to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to write processed tweets: {e}")
        return None


def read_processed_tweets(path: Path | str) -> List[ProcessedTweet]:
    if not pa:
        logger.error("PyArrow not installed, cannot read processed tweets.")
        return []
    try:
        path = Path(path)
        if not path.exists():
            return []

        table = pq.read_table(path)
        df = table.to_pandas()
        tweets = []
        for _, row in df.iterrows():
            tweets.append(ProcessedTweet(
                tweet_id=str(row.get('tweet_id', '')),
                username=row.get('username', ''),
                display_name=row.get('display_name', ''),
                timestamp=row.get('timestamp', None),
                content_raw=row.get('content_raw', ''),
                content_clean=row.get('content_clean', ''),
                lang=row.get('lang', 'en'),
                symbols=list(row.get('symbols', [])),
                hashtags=list(row.get('hashtags', [])),
                mentions=list(row.get('mentions', [])),
                likes=int(row.get('likes', 0)),
                retweets=int(row.get('retweets', 0)),
                replies=int(row.get('replies', 0)),
                views=int(row.get('views', 0)) if row.get('views') is not None else None,
                follower_count=int(row.get('follower_count', 0)),
                sentiment_score=float(row.get('sentiment_score', 0.0)),
                is_duplicate=bool(row.get('is_duplicate', False))
            ))
        return tweets
    except Exception as e:
        logger.error(f"Failed to read processed tweets from {path}: {e}")
        return []


def write_signals(signals: List[TradingSignal], path: Path | str = SIGNALS_DATA_DIR) -> Path | None:
    if not pa:
        logger.error("PyArrow not installed, cannot write signals.")
        return None
    if not signals:
        return None
    try:
        file_path = _resolve_target_path(path, "signals")
        data = {
            'timestamp': [s.timestamp for s in signals],
            'window_start': [s.window_start for s in signals],
            'window_end': [s.window_end for s in signals],
            'sentiment_mean': [float(s.sentiment_mean) for s in signals],
            'sentiment_ci_lower': [float(s.sentiment_ci_lower) for s in signals],
            'sentiment_ci_upper': [float(s.sentiment_ci_upper) for s in signals],
            'tweet_volume': [int(s.tweet_volume) for s in signals],
            'bull_bear_ratio': [float(s.bull_bear_ratio) for s in signals],
            'sentiment_velocity': [float(s.sentiment_velocity) for s in signals],
            'volume_anomaly_ratio': [float(s.volume_anomaly_ratio) for s in signals],
            'composite_score': [float(s.composite_score) for s in signals],
            'signal': [str(s.signal) for s in signals],
            'confidence': [float(s.confidence) for s in signals]
        }
        table = pa.Table.from_pydict(data)
        pq.write_table(table, file_path, compression='ZSTD')
        logger.info(f"Wrote {len(signals)} signals to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to write signals: {e}")
        return None


def read_signals(path: Path | str) -> List[TradingSignal]:
    if not pa:
        logger.error("PyArrow not installed, cannot read signals.")
        return []
    try:
        path = Path(path)
        if not path.exists():
            return []

        table = pq.read_table(path)
        df = table.to_pandas()
        signals = []
        for _, row in df.iterrows():
            signals.append(TradingSignal(
                timestamp=row.get('timestamp', None),
                window_start=row.get('window_start', None),
                window_end=row.get('window_end', None),
                sentiment_mean=float(row.get('sentiment_mean', 0.0)),
                sentiment_ci_lower=float(row.get('sentiment_ci_lower', 0.0)),
                sentiment_ci_upper=float(row.get('sentiment_ci_upper', 0.0)),
                tweet_volume=int(row.get('tweet_volume', 0)),
                bull_bear_ratio=float(row.get('bull_bear_ratio', 0.0)),
                sentiment_velocity=float(row.get('sentiment_velocity', 0.0)),
                volume_anomaly_ratio=float(row.get('volume_anomaly_ratio', 0.0)),
                composite_score=float(row.get('composite_score', 0.0)),
                signal=str(row.get('signal', 'HOLD')),
                confidence=float(row.get('confidence', 0.0))
            ))
        return signals
    except Exception as e:
        logger.error(f"Failed to read signals from {path}: {e}")
        return []

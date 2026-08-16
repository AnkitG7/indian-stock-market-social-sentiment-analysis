"""
CSV-based market data provider.
"""
from __future__ import annotations
import logging
import csv
from datetime import datetime
from pathlib import Path

from models import MarketCandle
from market_data.base_provider import MarketDataProvider

logger = logging.getLogger(__name__)

class CSVMarketDataProvider(MarketDataProvider):
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        
    def get_candles(self, symbol: str, start: datetime, end: datetime, interval_minutes: int = 5) -> list[MarketCandle]:
        candles = []
        if not self.csv_path.exists():
            logger.error(f"CSV file not found: {self.csv_path}")
            return candles
            
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_sym = row.get('symbol', '')
                    if row_sym != symbol:
                        continue
                        
                    ts_str = row.get('timestamp', '')
                    try:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    except ValueError:
                        try:
                            ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            continue
                            
                    if ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)
                        
                    if start <= ts <= end:
                        candles.append(MarketCandle(
                            timestamp=ts,
                            symbol=row_sym,
                            open=float(row.get('open', 0)),
                            high=float(row.get('high', 0)),
                            low=float(row.get('low', 0)),
                            close=float(row.get('close', 0)),
                            volume=int(float(row.get('volume', 0)))
                        ))
        except Exception as e:
            logger.error(f"Error reading CSV market data: {e}")
            
        return sorted(candles, key=lambda c: c.timestamp)

"""
Abstract market data provider interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime

from models import MarketCandle

class MarketDataProvider(ABC):
    @abstractmethod
    def get_candles(self, symbol: str, start: datetime, end: datetime, interval_minutes: int = 5) -> list[MarketCandle]:
        pass

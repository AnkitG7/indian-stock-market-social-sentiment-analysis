"""
Abstract base class for tweet collectors.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime

from models import RawTweet

logger = logging.getLogger(__name__)

class TweetCollector(ABC):
    """Abstract base class for collecting tweets from Twitter."""
    
    @abstractmethod
    async def collect(self, query: str, since: datetime, until: datetime, limit: int) -> list[RawTweet]:
        """Collect tweets matching the query."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the collector and release resources."""
        pass

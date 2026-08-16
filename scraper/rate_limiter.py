"""
Adaptive rate limiter for API requests.
"""
from __future__ import annotations

import asyncio
import logging
import random

logger = logging.getLogger(__name__)

class RateLimiter:
    """Adaptive rate limiter with exponential backoff and jitter."""
    
    def __init__(self, base_delay: float, max_delay: float):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.current_delay = base_delay
        self.failed_requests = 0
        
    async def wait(self) -> None:
        """Wait for the current delay amount with jitter."""
        jitter = random.uniform(0.8, 1.2)
        delay = self.current_delay * jitter
        logger.debug(f"RateLimiter waiting for {delay:.2f}s")
        await asyncio.sleep(delay)
        
    def on_success(self) -> None:
        """Reset delay to base delay upon success."""
        self.current_delay = self.base_delay
        
    def on_failure(self) -> None:
        """Double the delay up to max_delay upon failure."""
        self.failed_requests += 1
        self.current_delay = min(self.current_delay * 2, self.max_delay)
        logger.warning(f"Request failed. Increased delay to {self.current_delay:.2f}s")
        
    def on_rate_limit(self) -> None:
        """Set delay to max_delay upon encountering a rate limit."""
        self.failed_requests += 1
        self.current_delay = self.max_delay
        logger.warning(f"Rate limit encountered. Increased delay to {self.current_delay:.2f}s")

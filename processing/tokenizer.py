"""
Domain-specific financial tweet tokenizer.
"""
from __future__ import annotations

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Essential financial stopwords to retain
FINANCIAL_STOPWORDS_EXCLUSION = {
    'up', 'down', 'above', 'below', 'call', 'put', 'long', 'short',
    'gain', 'loss', 'high', 'low', 'buy', 'sell', 'bull', 'bear',
    'hold', 'target', 'stoploss', 'breakout', 'breakdown', 'not', 'no'
}

HINGLISH_SLANG_MAP = {
    'teji': 'bullish',
    'mandi': 'bearish',
    'sl hit': 'stoploss_hit',
    'phans gaye': 'trapped',
    'gap up': 'gap_up',
    'gap down': 'gap_down'
}

def financial_tweet_tokenizer(text: str) -> list[str]:
    """
    Tokenizer for TF-IDF. 
    Preserves specific tokens and maps Hinglish slang.
    """
    try:
        # Lowercase everything except special tokens which we'll handle later, 
        # or just lowercase all and keep tokens lowercase
        text_lower = text.lower()
        
        # Apply Hinglish slang mapping (multi-word replacements first)
        for slang, canonical in HINGLISH_SLANG_MAP.items():
            text_lower = text_lower.replace(slang, canonical)
            
        # Regex to capture words, special tokens like TICKER_BANKNIFTY, STRIKE_CE, EMOJI_BULL, $NIFTY
        # We can split on non-alphanumeric except for _, $, and maybe #
        tokens = re.findall(r'\$?[a-z0-9_]+', text_lower)
        
        result = []
        for token in tokens:
            # We don't filter stopwords here, TF-IDF will do it, but we want to ensure
            # we provide clean tokens.
            result.append(token)
            
        return result
    except Exception as e:
        logger.error(f"Error in tokenizer: {e}")
        return []

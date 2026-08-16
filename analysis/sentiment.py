"""
Hybrid sentiment engine for Indian Financial Market.
"""
from __future__ import annotations
import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            logger.info("Downloading NLTK vader_lexicon...")
            nltk.download('vader_lexicon', quiet=True)
            
        self.vader = SentimentIntensityAnalyzer()
        self._update_lexicon()
        
    def _update_lexicon(self):
        indian_financial_lexicon = {
            'teji': 2.5, 'bullish': 2.0, 'breakout': 2.2, 'rocket': 3.0, 
            'target_hit': 2.5, 'long_buildup': 2.0, 'gap_up': 1.8, 'jackpot': 2.8, 
            'recovery': 1.5, 'buying': 1.6, 'rally': 2.0, 'multibagger': 2.8, 
            'accumulate': 1.5, 'support': 1.0, 'EMOJI_BULL': 2.5,
            
            'mandi': -2.5, 'bearish': -2.0, 'breakdown': -2.2, 'tanking': -2.8, 
            'stoploss_hit': -2.0, 'trap': -2.4, 'phans_gaye': -2.5, 'gap_down': -1.8, 
            'crash': -3.2, 'loss': -2.0, 'selling': -1.6, 'profit_booking': -1.2, 
            'resistance': -1.0, 'EMOJI_BEAR': -2.5, 'panic': -2.5
        }
        self.vader.lexicon.update(indian_financial_lexicon)
        
    def analyze(self, text: str) -> float:
        try:
            if not text:
                return 0.0
            scores = self.vader.polarity_scores(text)
            return scores['compound']
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0
            
    def analyze_batch(self, texts: list[str]) -> list[float]:
        return [self.analyze(text) for text in texts]

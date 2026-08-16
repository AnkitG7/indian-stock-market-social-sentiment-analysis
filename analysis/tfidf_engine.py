"""
TF-IDF feature extraction engine with financial domain configuration.
"""
from __future__ import annotations

import logging
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import numpy as np

from config import config
from processing.tokenizer import financial_tweet_tokenizer

logger = logging.getLogger(__name__)


class TfidfFeatureEngine:
    """Domain-specific TF-IDF feature extractor for financial tweet n-grams."""

    def __init__(self, min_df: int | None = None) -> None:
        self.min_df = min_df if min_df is not None else config.analysis.tfidf_min_df
        self.vectorizer = TfidfVectorizer(
            tokenizer=financial_tweet_tokenizer,
            token_pattern=None,  # Suppress warning when tokenizer is provided
            ngram_range=config.analysis.tfidf_ngram_range,
            sublinear_tf=True,
            min_df=self.min_df,
            max_df=config.analysis.tfidf_max_df,
            max_features=config.analysis.tfidf_max_features,
        )
        self.feature_names = None

    def _ensure_compatible_min_df(self, n_docs: int) -> None:
        """Adjust min_df dynamically if document count is smaller than default min_df."""
        if n_docs < self.min_df:
            adjusted_min_df = max(1, n_docs)
            self.vectorizer = TfidfVectorizer(
                tokenizer=financial_tweet_tokenizer,
                token_pattern=None,
                ngram_range=config.analysis.tfidf_ngram_range,
                sublinear_tf=True,
                min_df=adjusted_min_df,
                max_df=1.0,
                max_features=config.analysis.tfidf_max_features,
            )

    def fit(self, texts: list[str]) -> None:
        try:
            if not texts:
                return
            self._ensure_compatible_min_df(len(texts))
            self.vectorizer.fit(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
        except Exception as e:
            logger.error(f"Error fitting TfidfVectorizer: {e}")

    def transform(self, texts: list[str]) -> Any:
        try:
            if not texts or self.feature_names is None:
                return scipy.sparse.csr_matrix((len(texts), 0))
            return self.vectorizer.transform(texts)
        except Exception as e:
            logger.error(f"Error transforming texts: {e}")
            return scipy.sparse.csr_matrix((len(texts), len(self.feature_names) if self.feature_names is not None else 0))

    def fit_transform(self, texts: list[str]) -> Any:
        try:
            if not texts:
                return scipy.sparse.csr_matrix((0, 0))
            self._ensure_compatible_min_df(len(texts))
            matrix = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            return matrix
        except Exception as e:
            logger.error(f"Error fit_transforming texts: {e}")
            return scipy.sparse.csr_matrix((len(texts), 0))

    def get_top_features(self, n: int = 20) -> list[tuple[str, float]]:
        if self.feature_names is None or not hasattr(self.vectorizer, 'idf_'):
            return []
        try:
            idf = self.vectorizer.idf_
            top_indices = np.argsort(idf)[::-1][:n]
            return [(self.feature_names[i], float(idf[i])) for i in top_indices]
        except Exception as e:
            logger.error(f"Error extracting top TF-IDF features: {e}")
            return []

    def get_window_features(self, texts: list[str], n: int = 10) -> list[tuple[str, float]]:
        if self.feature_names is None or not texts:
            return []
        try:
            matrix = self.transform(texts)
            if matrix.shape[0] == 0 or matrix.shape[1] == 0:
                return []
            mean_tfidf = np.asarray(matrix.mean(axis=0)).ravel()
            top_indices = np.argsort(mean_tfidf)[::-1][:n]
            return [(self.feature_names[i], float(mean_tfidf[i])) for i in top_indices if mean_tfidf[i] > 0]
        except Exception as e:
            logger.error(f"Error getting window features: {e}")
            return []

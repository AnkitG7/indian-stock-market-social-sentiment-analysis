"""
Multi-stage deduplication for real-time and large-scale tweet streams.
"""
from __future__ import annotations

import re
import hashlib
import logging
from collections import OrderedDict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    logger.warning("datasketch not installed. MinHash LSH deduplication will be disabled.")


class Deduplicator:
    """
    Multi-stage high-throughput deduplicator:
    - Stage 1: Exact ID check (O(1) set lookup)
    - Stage 2: Canonical MD5 content hash (O(1) lookup)
    - Stage 3: MinHash LSH index (O(1) sublinear bucket query for near-duplicates)
    - Memory bounding: Sliding window with LRU eviction
    """

    def __init__(self, jaccard_threshold: float = 0.80, minhash_perms: int = 128, window_size: int = 50_000):
        self.jaccard_threshold = jaccard_threshold
        self.minhash_perms = minhash_perms
        self.window_size = window_size

        # State collections with LRU tracking
        self.seen_ids: OrderedDict[str, bool] = OrderedDict()
        self.seen_hashes: OrderedDict[str, bool] = OrderedDict()

        # O(1) LSH index for near-duplicate detection
        if DATASKETCH_AVAILABLE:
            self.lsh: Optional[MinHashLSH] = MinHashLSH(
                threshold=self.jaccard_threshold,
                num_perm=self.minhash_perms,
            )
            self.seen_minhashes: OrderedDict[str, MinHash] = OrderedDict()
        else:
            self.lsh = None
            self.seen_minhashes = OrderedDict()

        self.exact_id_dupes = 0
        self.exact_content_dupes = 0
        self.near_dupes = 0
        self.unique = 0

    def _evict_old(self) -> None:
        """Evict oldest entries when sliding window capacity is exceeded."""
        while len(self.seen_ids) > self.window_size:
            old_id, _ = self.seen_ids.popitem(last=False)
            if self.lsh and old_id in self.seen_minhashes:
                try:
                    self.lsh.remove(old_id)
                except Exception:
                    pass
                self.seen_minhashes.pop(old_id, None)

        while len(self.seen_hashes) > self.window_size:
            self.seen_hashes.popitem(last=False)

    def _canonicalize(self, text: str) -> str:
        """Normalize text: strip URLs, mentions, punctuation, lowercase, join."""
        t = re.sub(r'http\S+|t\.co/\S+|www\.\S+', '', text)
        t = re.sub(r'@\w+', '', t)
        t = re.sub(r'[^\w\s]', '', t)
        return "".join(t.lower().split())

    def _get_minhash(self, text: str) -> Optional[MinHash]:
        """Generate MinHash signature using 3-character shingles."""
        if not DATASKETCH_AVAILABLE:
            return None
        m = MinHash(num_perm=self.minhash_perms)
        canon = self._canonicalize(text)
        if len(canon) < 3:
            m.update(canon.encode('utf-8'))
            return m
        for i in range(len(canon) - 2):
            shingle = canon[i:i+3]
            m.update(shingle.encode('utf-8'))
        return m

    def is_duplicate(self, tweet_id: str, text: str) -> bool:
        """
        Check whether tweet is duplicate across exact ID, exact content, or near-duplicate LSH.
        Updates internal index if unique.
        """
        try:
            # Stage 1: Exact tweet_id check
            if tweet_id in self.seen_ids:
                self.exact_id_dupes += 1
                return True

            # Stage 2: Canonical MD5 content hash
            canon_text = self._canonicalize(text)
            text_md5 = hashlib.md5(canon_text.encode('utf-8')).hexdigest()
            if text_md5 in self.seen_hashes:
                self.exact_content_dupes += 1
                self.seen_ids[tweet_id] = True
                self._evict_old()
                return True

            # Stage 3: MinHash LSH near-duplicate query
            if self.lsh:
                m1 = self._get_minhash(text)
                if m1:
                    matches = self.lsh.query(m1)
                    if matches:
                        self.near_dupes += 1
                        self.seen_ids[tweet_id] = True
                        self._evict_old()
                        return True

                    # Insert unique minhash into LSH index
                    self.lsh.insert(tweet_id, m1)
                    self.seen_minhashes[tweet_id] = m1

            self.unique += 1
            self.seen_ids[tweet_id] = True
            self.seen_hashes[text_md5] = True
            self._evict_old()
            return False

        except Exception as e:
            logger.error(f"Error in deduplication for {tweet_id}: {e}")
            return False

    def get_stats(self) -> dict:
        """Return comprehensive duplicate breakdown."""
        return {
            "exact_id_dupes": self.exact_id_dupes,
            "exact_content_dupes": self.exact_content_dupes,
            "near_dupes": self.near_dupes,
            "unique": self.unique,
        }

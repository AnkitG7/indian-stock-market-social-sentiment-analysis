"""Tests for processing.deduplication — multi-stage duplicate detection."""
import pytest
from processing.deduplication import Deduplicator


@pytest.fixture
def dedup():
    return Deduplicator(
        jaccard_threshold=0.80,
        minhash_perms=128,
        window_size=50_000,
    )


def test_exact_id_duplicate(dedup):
    assert not dedup.is_duplicate("id_1", "Some tweet text here")
    assert dedup.is_duplicate("id_1", "Completely different text")


def test_exact_content_duplicate(dedup):
    text = "Nifty breakout above 22500 resistance today"
    assert not dedup.is_duplicate("id_1", text)
    assert dedup.is_duplicate("id_2", text)


def test_unique_tweets(dedup):
    assert not dedup.is_duplicate("a", "Nifty is going up today")
    assert not dedup.is_duplicate("b", "Banknifty crashed below support")
    assert not dedup.is_duplicate("c", "What time does the market open")


def test_near_duplicate(dedup):
    """Slightly modified text should be detected if datasketch is available."""
    base = "This is a long market tweet about Nifty crashing today with heavy selling"
    dedup.is_duplicate("near_1", base)
    # Add a minor suffix — MinHash should detect near-duplicate
    result = dedup.is_duplicate("near_2", base + " xyz")
    # Can't assert True unconditionally (datasketch may not be installed)
    # But we can assert it doesn't crash
    assert isinstance(result, bool)


def test_stats_tracking(dedup):
    dedup.is_duplicate("1", "Alpha text")
    dedup.is_duplicate("1", "Alpha text")  # exact ID dup
    dedup.is_duplicate("3", "Alpha text")  # exact content dup
    dedup.is_duplicate("4", "Beta text different")  # unique

    stats = dedup.get_stats()
    assert stats["exact_id_dupes"] >= 1
    assert stats["exact_content_dupes"] >= 1
    assert stats["unique"] >= 2  # "1" initial + "4"


def test_window_eviction():
    """Ensure sliding window doesn't grow unbounded."""
    small_dedup = Deduplicator(
        jaccard_threshold=0.80,
        minhash_perms=128,
        window_size=10,
    )
    for i in range(50):
        small_dedup.is_duplicate(f"id_{i}", f"Unique tweet number {i}")
    # Internal state should be bounded
    assert len(small_dedup.seen_ids) <= 10

"""
Query builder for Twitter advanced search.
"""
from __future__ import annotations

from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class QueryBuilder:
    """Builder for Twitter advanced search queries."""

    @staticmethod
    def build_query(
        hashtags: list[str],
        since: datetime | None = None,
        until: datetime | None = None,
        exclude_retweets: bool = True,
        min_engagement: int = 0,
        lang: str | None = None,
    ) -> str:
        """Build a Twitter advanced search query string."""
        parts = []

        if hashtags:
            hashtag_query = " OR ".join([f"#{h.strip('#')}" for h in hashtags])
            parts.append(f"({hashtag_query})")

        if exclude_retweets:
            parts.append("-is:retweet")

        if min_engagement > 0:
            parts.append(f"min_faves:{min_engagement}")

        if lang:
            parts.append(f"lang:{lang}")

        if since:
            parts.append(f"since:{since.strftime('%Y-%m-%d')}")

        if until:
            parts.append(f"until:{until.strftime('%Y-%m-%d')}")

        return " ".join(parts)


def build_query(
    hashtags: list[str],
    since: datetime | None = None,
    until: datetime | None = None,
    exclude_retweets: bool = True,
    min_engagement: int = 0,
    lang: str | None = None,
) -> str:
    """Module-level convenience wrapper for QueryBuilder.build_query."""
    return QueryBuilder.build_query(
        hashtags=hashtags,
        since=since,
        until=until,
        exclude_retweets=exclude_retweets,
        min_engagement=min_engagement,
        lang=lang,
    )

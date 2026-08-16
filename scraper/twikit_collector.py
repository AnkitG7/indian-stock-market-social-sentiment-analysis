"""
Tweet collector using the twikit library.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

from twikit import Client
from twikit.errors import (
    TooManyRequests,
    TwitterException,
    ServerError,
    Unauthorized,
    Forbidden,
    AccountLocked,
)

from models import RawTweet
from config import config
from scraper.base_collector import TweetCollector
from scraper.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class TwikitCollector(TweetCollector):
    """Collects tweets using the Twikit library."""

    def __init__(self) -> None:
        self.client = Client('en-US')
        self.rate_limiter = RateLimiter(
            base_delay=config.scraping.base_delay,
            max_delay=config.scraping.max_delay,
        )
        self._authenticated = False

    async def _authenticate(self) -> None:
        """Authenticate using cookies or credentials."""
        if self._authenticated:
            return

        cookies_file = config.twitter.cookies_file
        try:
            if os.path.exists(cookies_file):
                self.client.load_cookies(cookies_file)
                logger.info("Authenticated using cookies.json file.")
                self._authenticated = True
                return

            if config.twitter.auth_token:
                cookie_dict = {"auth_token": config.twitter.auth_token}
                if config.twitter.ct0:
                    cookie_dict["ct0"] = config.twitter.ct0
                self.client.set_cookies(cookie_dict)
                logger.info("Authenticated using TWITTER_AUTH_TOKEN from .env.")
                self._authenticated = True
                return

            if config.twitter.username and config.twitter.password:
                user = config.twitter.username.lstrip('@')
                await self.client.login(
                    auth_info_1=user,
                    auth_info_2=config.twitter.email or user,
                    password=config.twitter.password,
                )
                self.client.save_cookies(cookies_file)
                logger.info("Authenticated using credentials and saved cookies.json.")
                self._authenticated = True
                return

            logger.warning("No Twitter credentials, auth_token, or cookies.json found. Twikit may be limited.")
            self._authenticated = True
        except Exception as e:
            logger.error(f"Failed to authenticate twikit: {e}")
            raise

    async def collect(self, query: str, since: datetime, until: datetime, limit: int) -> list[RawTweet]:
        """Collect tweets using Twikit."""
        await self._authenticate()

        logger.info(f"Starting Twikit collection for query: {query}")
        tweets_collected: list[RawTweet] = []

        try:
            await self.rate_limiter.wait()
            tweets = await self.client.search_tweet(query, product='Latest')
            self.rate_limiter.on_success()

            while tweets and len(tweets_collected) < limit:
                for tweet in tweets:
                    if len(tweets_collected) >= limit:
                        break

                    try:
                        # Extract and parse timestamp
                        if hasattr(tweet, 'created_at_datetime') and tweet.created_at_datetime:
                            created_at = tweet.created_at_datetime
                        else:
                            try:
                                created_at = datetime.strptime(str(tweet.created_at), "%a %b %d %H:%M:%S %z %Y")
                            except Exception:
                                created_at = datetime.fromisoformat(str(tweet.created_at).replace('Z', '+00:00'))

                        # Extract hashtags & mentions safely
                        raw_hashtags = getattr(tweet, 'hashtags', []) or []
                        hashtags = [
                            h if isinstance(h, str) else (h.get('text', '') if isinstance(h, dict) else str(h))
                            for h in raw_hashtags
                        ]

                        raw_mentions = getattr(tweet, 'user_mentions', []) or []
                        mentions = [
                            m if isinstance(m, str) else (m.get('screen_name', '') if isinstance(m, dict) else str(m))
                            for m in raw_mentions
                        ]

                        raw_tweet = RawTweet(
                            tweet_id=str(tweet.id),
                            username=str(tweet.user.screen_name if tweet.user else ""),
                            display_name=str(tweet.user.name if tweet.user else ""),
                            timestamp=created_at.replace(tzinfo=None) if created_at.tzinfo else created_at,
                            content=str(tweet.text or ""),
                            likes=int(getattr(tweet, 'favorite_count', 0) or 0),
                            retweets=int(getattr(tweet, 'retweet_count', 0) or 0),
                            replies=int(getattr(tweet, 'reply_count', 0) or 0),
                            views=int(getattr(tweet, 'view_count', 0) or 0),
                            hashtags=[h for h in hashtags if h],
                            mentions=[m for m in mentions if m],
                            follower_count=int(getattr(tweet.user, 'followers_count', 0) or 0) if tweet.user else 0,
                            url=f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}" if tweet.user else "",
                        )
                        tweets_collected.append(raw_tweet)
                    except Exception as e:
                        logger.warning(f"Error mapping tweet {getattr(tweet, 'id', 'unknown')}: {e}")

                if len(tweets_collected) >= limit:
                    break

                retries = 0
                max_retries = config.scraping.max_retries

                while retries < max_retries:
                    try:
                        logger.info(f"Collected {len(tweets_collected)} tweets so far. Fetching next page...")
                        await self.rate_limiter.wait()
                        tweets = await tweets.next()
                        self.rate_limiter.on_success()
                        break
                    except TooManyRequests:
                        logger.warning("Rate limit hit during pagination.")
                        self.rate_limiter.on_rate_limit()
                        retries += 1
                    except (TwitterException, ServerError) as e:
                        logger.warning(f"Error during pagination: {e}")
                        self.rate_limiter.on_failure()
                        retries += 1
                    except Exception as e:
                        logger.error(f"Unexpected error during pagination: {e}")
                        self.rate_limiter.on_failure()
                        retries += 1

                if retries >= max_retries:
                    logger.error("Max retries reached during pagination.")
                    break

        except TooManyRequests:
            logger.warning("Rate limit hit during initial search.")
            self.rate_limiter.on_rate_limit()
        except Exception as e:
            logger.error(f"Error during Twikit collection: {e}")

        logger.info(f"Twikit collection complete. Gathered {len(tweets_collected)} tweets.")
        return tweets_collected

    async def close(self) -> None:
        """Close resources."""
        pass

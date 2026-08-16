"""
Fallback tweet collector using Selenium (undetected-chromedriver).
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import json
import os
import subprocess
from datetime import datetime
from urllib.parse import quote

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException

from models import RawTweet
from config import config
from scraper.base_collector import TweetCollector

logger = logging.getLogger(__name__)


def _detect_chrome_major_version() -> int | None:
    """Auto-detect installed Google Chrome major version on Windows."""
    try:
        paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ]
        for p in paths:
            if os.path.exists(p):
                cmd = f'(Get-Item "{p}").VersionInfo.FileVersion'
                res = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd]).decode().strip()
                major = int(res.split('.')[0])
                logger.info(f"Detected Chrome major version: {major} (Full: {res})")
                return major
    except Exception as e:
        logger.warning(f"Could not auto-detect Chrome version: {e}")
    return None


class SeleniumCollector(TweetCollector):
    """Collects tweets using undetected-chromedriver as a robust fallback."""

    def __init__(self) -> None:
        self.driver = None
        self._authenticated = False

    async def _init_driver(self) -> None:
        """Initialize undetected-chromedriver with matching version and headless settings."""
        if self.driver is None:
            options = uc.ChromeOptions()
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--log-level=3")

            chrome_ver = _detect_chrome_major_version()
            loop = asyncio.get_running_loop()

            def _create_driver():
                kwargs = {"options": options}
                if chrome_ver:
                    kwargs["version_main"] = chrome_ver
                return uc.Chrome(**kwargs)

            self.driver = await loop.run_in_executor(None, _create_driver)
            self.driver.set_page_load_timeout(45)
            logger.info("Undetected ChromeDriver initialized successfully.")

    async def _authenticate(self) -> None:
        """Inject cookies to authenticate if available."""
        if self._authenticated:
            return

        await self._init_driver()
        cookies_file = config.twitter.cookies_file
        loop = asyncio.get_running_loop()

        def _login():
            try:
                self.driver.get("https://x.com")
                time_sleep = 3
                if os.path.exists(cookies_file):
                    with open(cookies_file, 'r', encoding='utf-8') as f:
                        cookies = json.load(f)
                    if isinstance(cookies, dict):
                        for k, v in cookies.items():
                            for d in ['.x.com', '.twitter.com']:
                                try:
                                    self.driver.add_cookie({'name': k, 'value': v, 'domain': d, 'path': '/'})
                                except Exception:
                                    pass
                    else:
                        for cookie in cookies:
                            cookie_dict = {
                                'name': cookie['name'],
                                'value': cookie['value'],
                                'domain': cookie.get('domain', '.x.com'),
                                'path': cookie.get('path', '/'),
                            }
                            try:
                                self.driver.add_cookie(cookie_dict)
                            except Exception:
                                pass
                    logger.info("Selenium authenticated via cookies.json.")
                elif config.twitter.auth_token:
                    for d in ['.x.com', '.twitter.com']:
                        try:
                            self.driver.add_cookie({
                                'name': 'auth_token',
                                'value': config.twitter.auth_token,
                                'domain': d,
                                'path': '/',
                            })
                            if config.twitter.ct0:
                                self.driver.add_cookie({
                                    'name': 'ct0',
                                    'value': config.twitter.ct0,
                                    'domain': d,
                                    'path': '/',
                                })
                        except Exception:
                            pass
                    logger.info("Selenium authenticated via TWITTER_AUTH_TOKEN.")
                else:
                    logger.warning("No cookies file or auth_token found for Selenium.")
            except Exception as e:
                logger.error(f"Error during Selenium authentication: {e}")

        await loop.run_in_executor(None, _login)
        self._authenticated = True

    def _parse_count(self, text: str) -> int:
        """Parse engagement count string (e.g. '1.5K' -> 1500, '2M' -> 2000000)."""
        if not text:
            return 0
        text = text.upper().replace(',', '').strip()
        try:
            if 'K' in text:
                return int(float(text.replace('K', '')) * 1000)
            if 'M' in text:
                return int(float(text.replace('M', '')) * 1000000)
            return int(float(text))
        except ValueError:
            return 0

    async def collect(self, query: str, since: datetime, until: datetime, limit: int) -> list[RawTweet]:
        """Collect tweets using Selenium with incremental DOM extraction and virtual scroll handling."""
        await self._authenticate()

        encoded_query = quote(query)
        search_url = f"https://x.com/search?q={encoded_query}&src=typed_query&f=live"
        logger.info(f"Selenium navigating to search: {search_url}")

        loop = asyncio.get_running_loop()

        def _navigate():
            self.driver.get(search_url)
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'article[data-testid="tweet"]'))
                )
            except TimeoutException:
                logger.warning("Timeout waiting for tweet elements to load on Twitter/X.")

        await loop.run_in_executor(None, _navigate)

        tweets_collected: list[RawTweet] = []
        seen_ids: set[str] = set()

        scroll_attempts = 0
        max_scroll_attempts = max(limit * 3, 50)
        no_new_tweets_count = 0

        while len(tweets_collected) < limit and scroll_attempts < max_scroll_attempts:
            def _extract_tweets():
                elements = self.driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
                extracted = []
                for element in elements:
                    try:
                        time_el = element.find_element(By.CSS_SELECTOR, 'time')
                        timestamp_str = time_el.get_attribute('datetime')

                        link_el = time_el.find_element(By.XPATH, '..')
                        tweet_url = link_el.get_attribute('href') or ""
                        tweet_id = tweet_url.split('/status/')[-1].split('?')[0] if '/status/' in tweet_url else None

                        if not tweet_id or tweet_id in seen_ids:
                            continue

                        user_info_el = element.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"]')
                        user_text = user_info_el.text.split('\n')
                        display_name = user_text[0] if len(user_text) > 0 else ""
                        username = user_text[1].replace('@', '') if len(user_text) > 1 else ""

                        try:
                            content_el = element.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
                            content = content_el.text
                        except NoSuchElementException:
                            content = ""

                        likes, retweets, replies, views = 0, 0, 0, 0
                        try:
                            group_el = element.find_element(By.CSS_SELECTOR, '[role="group"]')
                            aria_label = group_el.get_attribute('aria-label') or ""

                            replies_match = re.search(r'([\d\.,KM]+)\s*repl', aria_label, re.IGNORECASE)
                            retweets_match = re.search(r'([\d\.,KM]+)\s*repost', aria_label, re.IGNORECASE)
                            likes_match = re.search(r'([\d\.,KM]+)\s*like', aria_label, re.IGNORECASE)
                            views_match = re.search(r'([\d\.,KM]+)\s*view', aria_label, re.IGNORECASE)

                            if replies_match:
                                replies = self._parse_count(replies_match.group(1))
                            if retweets_match:
                                retweets = self._parse_count(retweets_match.group(1))
                            if likes_match:
                                likes = self._parse_count(likes_match.group(1))
                            if views_match:
                                views = self._parse_count(views_match.group(1))
                        except NoSuchElementException:
                            pass

                        hashtags = re.findall(r'#(\w+)', content)
                        mentions = re.findall(r'@(\w+)', content)

                        try:
                            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).replace(tzinfo=None)
                        except Exception:
                            dt = datetime.now()

                        raw_tweet = RawTweet(
                            tweet_id=tweet_id,
                            username=username,
                            display_name=display_name,
                            timestamp=dt,
                            content=content,
                            likes=likes,
                            retweets=retweets,
                            replies=replies,
                            views=views,
                            hashtags=hashtags,
                            mentions=mentions,
                            follower_count=0,
                            url=tweet_url,
                        )

                        extracted.append(raw_tweet)
                        seen_ids.add(tweet_id)

                    except Exception as e:
                        logger.debug(f"Failed to parse a tweet element: {e}")

                return extracted

            new_tweets = await loop.run_in_executor(None, _extract_tweets)
            if new_tweets:
                tweets_collected.extend(new_tweets)
                no_new_tweets_count = 0
                if len(tweets_collected) % 25 == 0 or len(tweets_collected) >= limit:
                    logger.info(f"Collected {len(tweets_collected)}/{limit} real tweets...")
            else:
                no_new_tweets_count += 1
                if no_new_tweets_count > 15:
                    logger.info(f"Reached bottom of search stream with {len(tweets_collected)} tweets.")
                    break

            if len(tweets_collected) >= limit:
                break

            def _scroll():
                self.driver.execute_script("window.scrollBy(0, window.innerHeight * 0.90);")

            await loop.run_in_executor(None, _scroll)
            scroll_attempts += 1

            delay = random.uniform(1.2, 2.2)
            await asyncio.sleep(delay)

        logger.info(f"Selenium collection complete. Gathered {len(tweets_collected[:limit])} real tweets.")
        return tweets_collected[:limit]

    async def close(self) -> None:
        """Safely close Selenium WebDriver."""
        if self.driver:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, self.driver.quit)
            except Exception:
                pass
            self.driver = None
            logger.info("Selenium WebDriver closed.")

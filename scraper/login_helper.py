"""
Automated headless & interactive login helper for Twitter/X using Selenium.
Logs in with credentials from .env and exports cookies.json automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import logging

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

from config import config
from scraper.selenium_collector import _detect_chrome_major_version

logger = logging.getLogger(__name__)


def auto_login_and_save_cookies(headless: bool = True) -> bool:
    """
    Automate Twitter/X login flow using undetected-chromedriver.
    Enters username, handles email confirmation prompt if asked, enters password,
    and captures all session cookies into cookies.json.
    """
    username = config.twitter.username.lstrip('@')
    email = config.twitter.email
    password = config.twitter.password

    if not username or not password:
        logger.error("No TWITTER_USERNAME or TWITTER_PASSWORD in .env")
        return False

    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1280,800")
    options.add_argument("--log-level=3")

    chrome_ver = _detect_chrome_major_version()
    kwargs = {"options": options}
    if chrome_ver:
        kwargs["version_main"] = chrome_ver

    print(f"[*] Initializing Chrome (version {chrome_ver})...")
    driver = uc.Chrome(**kwargs)

    try:
        print("[*] Navigating to Twitter/X login page...")
        driver.get("https://twitter.com/i/flow/login")
        time.sleep(5)

        # 1. Enter Username / Email
        print(f"[*] Entering username: {username}")
        username_el = WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.XPATH, "//input[@autocomplete='username' or @name='text']"))
        )
        username_el.clear()
        username_el.send_keys(username)
        username_el.send_keys(Keys.RETURN)
        time.sleep(5)

        # 2. Check for intermediate screen (unusual activity / email challenge)
        try:
            challenge_els = driver.find_elements(By.XPATH, "//input[@data-testid='ocfEnterTextTextInput' or @name='text']")
            for ch in challenge_els:
                if ch.is_displayed():
                    print(f"[*] Twitter requested confirmation. Entering email: {email}")
                    ch.clear()
                    ch.send_keys(email)
                    ch.send_keys(Keys.RETURN)
                    time.sleep(5)
                    break
        except Exception:
            pass

        # 3. Enter Password
        print("[*] Entering password...")
        password_el = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//input[@type='password' or @name='password']"))
        )
        password_el.clear()
        password_el.send_keys(password)
        password_el.send_keys(Keys.RETURN)
        time.sleep(10)

        print(f"[*] Login flow submitted. Current URL: {driver.current_url}")
        cookies = driver.get_cookies()
        auth_cookie = next((c for c in cookies if c['name'] == 'auth_token'), None)

        if auth_cookie:
            cookies_path = Path(config.twitter.cookies_file)
            with open(cookies_path, "w", encoding="utf-8") as f:
                json.dump(cookies, f, indent=2)
            print(f"[SUCCESS] Login authenticated! Captured {len(cookies)} cookies (auth_token verified).")
            print(f"[SUCCESS] Saved to {cookies_path.name}")
            return True
        else:
            print("[INFO] Checking if Twitter is on 2FA or home feed...")
            print(f"Page title: {driver.title}")
            if cookies:
                cookies_path = Path(config.twitter.cookies_file)
                with open(cookies_path, "w", encoding="utf-8") as f:
                    json.dump(cookies, f, indent=2)
            return False

    except Exception as e:
        print(f"[ERROR] Automated login error: {e}")
        return False
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    auto_login_and_save_cookies(headless=True)

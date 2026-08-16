# Code Architecture, Technical Mechanics & Design Decisions
**Indian Stock Market Tweet Intelligence & Quantitative Signal System**

---

## 📖 Table of Contents
1. [Executive Overview](#1-executive-overview)
2. [End-to-End Code Working Flow](#2-end-to-end-code-working-flow)
3. [Technology Stack: What Was Used and Why](#3-technology-stack-what-was-used-and-why)
4. [Deep Architectural Trade-Offs: "Why This Way, Not That Way"](#4-deep-architectural-trade-offs-why-this-way-not-that-way)
5. [Layer-by-Layer Implementation Mechanics](#5-layer-by-layer-implementation-mechanics)
6. [Mathematical Derivations & Statistical Rigor](#6-mathematical-derivations--statistical-rigor)
7. [System Error Handling & Edge Cases](#7-system-error-handling--edge-cases)

---

## 1. Executive Overview

This codebase is an **asynchronous, memory-bounded, modular quantitative pipeline** designed to collect, normalize, extract domain features, and generate statistically gated trading signals from live Indian stock market social conversations (`#nifty50`, `#sensex`, `#banknifty`, `#intraday`) without relying on paid APIs.

The system is decoupled into five distinct, independent layers:
```text
Ingestion (Scraper) ──► Normalization (Processing) ──► NLP & Feature Extraction (Analysis)
                                                                 │
Visual Dashboards ◄── Structured Reports ◄── Market Backtesting ◄┘
```

---

## 2. End-to-End Code Working Flow

When `python main.py run` is invoked, the execution flows sequentially through 6 automated stages:

```mermaid
flowchart TD
    subgraph S1["Stage 1: Ingestion (scraper/)"]
        Q["QueryBuilder: (#nifty50 OR #banknifty...)"] --> TC["TwikitCollector (GraphQL)"]
        TC -- "On Rate Limit / Failure" --> SC["SeleniumCollector (undetected-chromedriver)"]
        TC --> RL["RateLimiter (Backoff + Jitter)"]
        SC --> RL
        RL --> RAW["RawTweet Objects (2,000 Tweets)"]
        RAW --> W_RAW["write_raw_tweets() ──► Parquet + ZSTD"]
    end

    subgraph S2["Stage 2: Processing (processing/)"]
        W_RAW --> CLN["clean_tweet(): Strip URLs, normalizes @mentions, emojis, strikes, Hinglish"]
        CLN --> DEDUP["Deduplicator: Stage 1 (ID) ──► Stage 2 (MD5) ──► Stage 3 (MinHash LSH)"]
        DEDUP --> PROC["ProcessedTweet Objects (1,786 Unique Tweets)"]
        PROC --> W_PROC["write_processed_tweets() ──► Parquet"]
    end

    subgraph S3["Stage 3: Statistical NLP & TF-IDF (analysis/)"]
        W_PROC --> SA["SentimentAnalyzer: VADER + 50+ Indian Financial Lexicon"]
        W_PROC --> TFIDF["TfidfFeatureEngine: Domain Tokenizer + Stopword Exclusion"]
    end

    subgraph S4["Stage 4: Quantitative Signals (analysis/signal_generator.py)"]
        SA & TFIDF --> SIG["SignalGenerator: 15m Windows"]
        SIG --> MATH["• Engagement Weight w_i\n• Exponential Decay omega_i(t)\n• Kish Effective N_eff\n• Weighted SEM 95% CI\n• Volume Anomaly Ratio VAR >= 1.50"]
        MATH --> TSIG["TradingSignal Objects (BUY / SELL / HOLD)"]
        TSIG --> W_SIG["write_signals() ──► Parquet"]
    end

    subgraph S5["Stage 5: Validation & Visuals (analysis/)"]
        W_SIG & OHLCV["Historical 5m OHLCV Candles"] --> MV["MarketValidator: Forward 15m Price Return"]
        W_SIG & PROC --> VIS["Visualizer: LTTB Downsampling ──► 7 Publication PNGs"]
    end

    subgraph S6["Stage 6: Structured Reporting (main.py + advanced_analytics.py)"]
        MV & VIS & TSIG --> REP["output/reports/\n• signals_detailed.csv\n• market_intelligence_report.md\n• executive_summary.json\n• advanced_quant_insights.md"]
    end
```

---

## 3. Technology Stack: What Was Used and Why

| Technology / Library | Purpose in Code | Why It Was Chosen |
| :--- | :--- | :--- |
| **Python 3.11 + `@dataclass`** | Core language & domain models | Provides strong type hinting, minimal memory overhead ($O(1)$ attribute lookup), and native serialization without heavy ORM bloat. |
| **PyArrow + Parquet (ZSTD)** | Data storage (`data/raw`, `data/processed`, `data/signals`) | Significant columnar storage reduction and fast typed reads compared to raw CSV/JSON. Enables columnar dictionary encoding for low-cardinality columns (`lang`, `symbols`) and fast batch reads. |
| **VADER + Custom Lexicon** | Financial sentiment analysis | **Sub-millisecond CPU execution**, zero GPU/PyTorch dependencies, and exact deterministic scoring for Indian market slang (`teji`, `mandi`, `multibagger`, `breakout`). |
| **scikit-learn (`TfidfVectorizer`)** | N-gram feature extraction | Fast, C-optimized sparse matrix extraction. Integrated with custom financial tokenizers that prevent crucial market words from being removed. |
| **`datasketch` (MinHash LSH)** | Fuzzy near-duplicate deduplication | Provides approximately sublinear candidate lookup for near-duplicate detection, avoiding the $O(N^2)$ pairwise comparison approach. |
| **`undetected-chromedriver`** | Fallback browser collection | Browser-automation approach for authenticated collection with humanized delays and cookie injection; does not rely on paid API subscriptions. |
| **`twikit`** | Primary async GraphQL scraper | Direct asynchronous GraphQL API query collection with low latency and session cookie persistence. |
| **`matplotlib` + LTTB** | Visualization & plotting engine | Clean vector/raster plots with **Largest-Triangle-Three-Buckets downsampling** to render high-density charts without UI/RAM freezing. |
| **`pytest` + `tracemalloc`** | Testing & scalability verification | Comprehensive test runner combined with Python standard library memory tracing for accurate RAM benchmarks. |

---

## 4. Deep Architectural Trade-Offs: "Why This Way, Not That Way"

### Decision 1: `TweetCollector` Abstract Base Class vs. Direct Scraping Calls
* **Way Chosen**: An abstract base class [`TweetCollector(ABC)`](file:///c:/FS/QodeAdvREL/scraper/base_collector.py) defines the contract: `collect(query, since, until, limit) -> list[RawTweet]`. Both `TwikitCollector` and `SeleniumCollector` implement this interface.
* **Why NOT the other way**: Direct coupling to Selenium or Twikit makes the entire pipeline brittle. If Twitter changes its DOM or GraphQL endpoint, only the collector implementation is updated or swapped—the processing, deduplication, NLP, and signal layers remain completely unaffected.

---

### Decision 2: VADER + Indian Financial Lexicon vs. Heavy Transformers (e.g. FinBERT)
* **Way Chosen**: Rule-augmented, domain-customized VADER scoring running on CPU with 50+ Hindi/Hinglish financial terms (`teji`, `mandi`, `breakout`, `multibagger`, `stoploss_hit`, `jackpot`, `strike_ce`, `strike_pe`).
* **Why NOT the other way**:
  1. *Resource Constraints*: FinBERT requires a 500 MB+ model download, PyTorch/CUDA runtime, and 4-8 GB of RAM.
  2. *Throughput*: VADER processes **>2,500 tweets per second on CPU**, whereas FinBERT on CPU processes ~10-20 tweets/s (causing a 2-minute pipeline to take 25+ minutes).
  3. *Domain Mismatch*: Off-the-shelf FinBERT was trained on Wall Street corporate filings (10-Ks, Reuters transcripts) and completely fails on Indian retail market slang like *"Bhai 22500 call me teji aayegi rocket 🚀"*.

---

### Decision 3: 3-Stage Multi-Pass Deduplication vs. Database Unique Constraints
* **Way Chosen**: A 3-stage funnel:
  1. Exact Tweet ID lookup (hash set, $O(1)$).
  2. Canonical text normalizer $\to$ MD5 hash set ($O(1)$).
  3. MinHash LSH with 128 permutations across 3-character shingles ($J \ge 0.80$) bounded by a 10,000-item LRU sliding window.
* **Why NOT the other way**:
  - Simple database constraints only check exact primary keys.
  - Spammers and bot networks frequently post the exact same promotional text under different tweet IDs, or append random character tags (e.g. `"...call le lo #nifty [ref:8492]"`).
  - The 3-stage funnel catches all three types: 109 exact duplicates and 105 near-duplicate spam variations were successfully removed from the 2,000 raw tweets.

---

### Decision 4: Kish Effective Sample Size ($N_{\text{eff}}$) & Weighted SEM vs. Simple Standard Error
* **Way Chosen**:
  $$N_{\text{eff}} = \frac{\left(\sum \omega_i\right)^2}{\sum \omega_i^2}, \quad s_w = \sqrt{\frac{\sum \omega_i(S_i - \mu_t)^2}{\sum \omega_i}}, \quad \text{CI}_{95\%} = \mu_t \pm 1.96 \cdot \frac{s_w}{\sqrt{N_{\text{eff}}}}$$
* **Why NOT the other way**:
  - Naive Standard Error of the Mean ($\text{SEM} = \frac{\sigma}{\sqrt{N}}$) assumes unweighted, independent, identically distributed (i.i.d.) observations.
  - In financial social media, an institutional analyst tweet with 50,000 followers and 500 retweets carries far greater weight ($w_i$) than a zero-follower bot.
  - Using unweighted SEM underestimates variance when a single high-weight tweet dominates the window. Kish's $N_{\text{eff}}$ adjusts the degrees of freedom downward to reflect the true concentration of influence.

---

### Decision 5: Gated Composite Signals vs. Raw Sentiment Thresholds
* **Way Chosen**: A trade signal (`BUY`/`SELL`) is ONLY generated if:
  1. $\text{CI}_{\text{lower}} > +0.20$ (for BUY) or $\text{CI}_{\text{upper}} < -0.20$ (for SELL).
  2. **AND** $\text{Volume Anomaly Ratio} = \frac{\text{window\_volume}}{\text{rolling\_mean}} \ge 1.50$.
* **Why NOT the other way**:
  - If a signal is generated whenever average sentiment is slightly positive (e.g. $+0.05$), the system generates dozens of false-positive trades during low-volume, noisy chop.
  - Requiring the entire 95% Confidence Interval to clear $\pm 0.20$ ensures **statistical certainty**, and requiring $\text{VAR} \ge 1.50$ ensures **market participation momentum**.
  - When the market is diffuse, returning `HOLD` is the mathematically correct and risk-managed behavior.

---

### Decision 6: Largest-Triangle-Three-Buckets (LTTB) Downsampling vs. Uniform Sampling
* **Way Chosen**: Downsamples high-frequency sentiment and price points to a maximum of 1,500 points using the LTTB geometric triangle-area algorithm.
* **Why NOT the other way**:
  - Uniform sampling (`data[::10]`) drops 9 out of 10 data points arbitrarily, often missing extreme intraday spikes, crash bottoms, and volatility breakouts.
  - LTTB splits data into temporal buckets and selects the point that maximizes the triangular area with adjacent buckets, mathematically preserving the visual envelope, peak highs, and trough lows.

---

### Decision 7: Parquet + ZSTD vs. SQLite or Flat CSV
* **Way Chosen**: Columnar Apache Parquet files compressed with Zstandard (`ZSTD`).
* **Why NOT the other way**:
  - Flat CSV files parse strings on every read, consume 8-10× more disk space, and have no native timestamp or list schema support.
  - SQLite introduces disk write-lock contention in concurrent async pipelines.
  - Parquet allows reading only required columns (e.g. `sentiment_score` and `timestamp`) without loading tweet text into memory, keeping RAM usage below 3 MB.

---

## 5. Layer-by-Layer Implementation Mechanics

### Layer 1: Ingestion (`scraper/`)
1. [`query_builder.py`](file:///c:/FS/QodeAdvREL/scraper/query_builder.py): Combines target hashtags into boolean search strings:
   `(#nifty50 OR #sensex OR #intraday OR #banknifty) -is:retweet`
2. [`rate_limiter.py`](file:///c:/FS/QodeAdvREL/scraper/rate_limiter.py): Manages request pacing. On HTTP 429 or connection lag, it doubles backoff delay with randomized jitter:
   $$\text{delay} = \min(\text{base} \cdot 2^{\text{failures}} + \text{uniform}(0, 1), \text{max\_delay})$$
3. [`twikit_collector.py`](file:///c:/FS/QodeAdvREL/scraper/twikit_collector.py): Executes GraphQL search queries asynchronously with cookie session resumption.
4. [`selenium_collector.py`](file:///c:/FS/QodeAdvREL/scraper/selenium_collector.py): Auto-detects local Chrome major version (`v151`), initializes `undetected-chromedriver` in headless mode, injects `auth_token` cookies, and extracts tweet DOM cards (`article[data-testid="tweet"]`) with humanized scroll intervals.

---

### Layer 2: Processing & Storage (`processing/`)
1. [`cleaner.py`](file:///c:/FS/QodeAdvREL/processing/cleaner.py):
   - Decodes HTML entities (`&amp;` $\to$ `&`).
   - Removes HTTP/HTTPS URLs and Telegram links.
   - Normalizes user mentions (`@user` $\to$ `TOKEN_MENTION`).
   - Maps market emojis: `🚀🐂📈🟢` $\to$ `EMOJI_BULL`, `🐻🩸📉🔴` $\to$ `EMOJI_BEAR`.
   - Normalizes cashtags and symbols: `$NIFTY`, `#BankNifty` $\to$ `TICKER_BANKNIFTY`.
   - Extracts option strikes: `22500CE` $\to$ `STRIKE_CE`, `48500PE` $\to$ `STRIKE_PE`.
   - Detects language script: Devanagari Hindi (`hi`), Hinglish phonetic (`hinglish`), English (`en`).
2. [`tokenizer.py`](file:///c:/FS/QodeAdvREL/processing/tokenizer.py):
   - Custom financial tokenizer that retains financial stopwords (`up`, `down`, `call`, `put`, `long`, `short`, `buy`, `sell`, `breakout`, `support`, `resistance`).
3. [`deduplication.py`](file:///c:/FS/QodeAdvREL/processing/deduplication.py):
   - Evaluates tweet stream through ID set $\to$ MD5 hash $\to$ MinHash LSH tables.
4. [`storage.py`](file:///c:/FS/QodeAdvREL/processing/storage.py):
   - Serializes dataclasses into PyArrow tables matching `TWEET_SCHEMA` with ZSTD compression level 3.

---

### Layer 3: NLP & Analysis (`analysis/`)
1. [`sentiment.py`](file:///c:/FS/QodeAdvREL/analysis/sentiment.py):
   - Updates NLTK VADER lexicon with calibrated financial valence scores:
     - Bullish: `teji` (+2.5), `breakout` (+2.2), `multibagger` (+2.8), `target_hit` (+2.5), `EMOJI_BULL` (+2.5).
     - Bearish: `mandi` (-2.5), `breakdown` (-2.2), `stoploss_hit` (-2.0), `crash` (-3.2), `EMOJI_BEAR` (-2.5).
2. [`tfidf_engine.py`](file:///c:/FS/QodeAdvREL/analysis/tfidf_engine.py):
   - Fits `TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, min_df=2)` using `financial_tweet_tokenizer`.

---

### Layer 4: Signals & Validation (`analysis/`)
1. [`signal_generator.py`](file:///c:/FS/QodeAdvREL/analysis/signal_generator.py):
   - Segments tweets into rolling 15-minute time windows.
   - Calculates dynamic weights, half-life time decay, Kish effective sample size, weighted variance, and 95% Confidence Intervals.
2. [`market_validator.py`](file:///c:/FS/QodeAdvREL/analysis/market_validator.py):
   - Reads historical 5-minute OHLCV candles (`sample_output/sample_market_data.csv`).
   - For every signal at timestamp $t$, finds candle close at $t$ and candle close at $t + 15\text{m}$.
   - Evaluates directional correctness:
     $$\text{Forward Return} = \frac{P_{t+15\text{m}} - P_t}{P_t}$$
     - `BUY` is correct if $\text{Return} > 0$.
     - `SELL` is correct if $\text{Return} < 0$.

---

### Layer 5: Visualizations & Reporting (`analysis/visualizer.py`, `main.py`)
1. **Visual Dashboards**:
   - `sentiment_timeline.png`: LTTB downsampled scatter & rolling mean.
   - `sentiment_candlesticks.png`: Windowed sentiment OHLCV aggregation with 95% CI error bars.
   - `volume_heatmap.png`: Hourly volume intensity grid.
   - `signal_dashboard.png`: Composite score time series with BUY/SELL threshold bands.
   - `top_tfidf_features.png`: Horizontal bar chart of top financial n-grams.
   - `advanced_analytics_dashboard.png`: Social PCR strike distribution and influencer tier spreads.
2. **Structured Reports**:
   - `market_intelligence_report.md`: Executive summary with stock/index rankings and sentiment breakdown.
   - `advanced_quant_insights.md`: Social PCR, tier divergence, and market regime analysis.
   - `signals_detailed.csv`: 15-minute signal time series with 95% CI bounds.
   - `executive_summary.json`: Machine-readable metrics payload.

---

## 6. Mathematical Derivations & Statistical Rigor

### Logarithmic Engagement Weighting
To prevent virality outliers from dominating sentiment calculations, author followers and engagement metrics are scaled logarithmically:
$$w_i = \ln\left(1 + \text{likes}_i + 2 \cdot \text{retweets}_i + 0.5 \cdot \text{replies}_i\right) \cdot \ln\left(1 + \text{followers}_i\right)$$

### Half-Life Exponential Temporal Attenuation
A tweet posted at the start of a time window carries less predictive weight than a tweet posted immediately prior to the window close:
$$\lambda = \frac{\ln 2}{t_{\text{half-life}}}, \quad \omega_i(t) = w_i \cdot \exp\left(-\lambda \cdot (t_{\text{window\_end}} - t_i)\right)$$

### Weighted Sentiment Mean & Variance
$$\mu_t = \frac{\sum_{i=1}^N \omega_i \cdot S_i}{\sum_{i=1}^N \omega_i}$$
$$s_w^2 = \frac{\sum_{i=1}^N \omega_i \cdot (S_i - \mu_t)^2}{\sum_{i=1}^N \omega_i}$$

### Kish's Effective Sample Size ($N_{\text{eff}}$) & 95% Confidence Interval
When observations carry unequal weights, the effective sample size is derived using Kish's formula (1965):
$$N_{\text{eff}} = \frac{\left(\sum_{i=1}^N \omega_i\right)^2}{\sum_{i=1}^N \omega_i^2}$$
$$\text{SEM}_w = \frac{s_w}{\sqrt{N_{\text{eff}}}}$$
$$\text{CI}_{95\%} = \left[\mu_t - 1.96 \cdot \text{SEM}_w, \; \mu_t + 1.96 \cdot \text{SEM}_w\right]$$

---

## 7. System Error Handling & Edge Cases

| Scenario / Edge Case | How the Code Handles It |
| :--- | :--- |
| **Twikit Rate-Limited or Blocked** | `main.py` catches the exception and automatically engages fallback to `SeleniumCollector` with zero user intervention. |
| **Chrome Version Mismatch** | `scraper/selenium_collector.py` auto-queries Windows Registry and Powershell for the installed Chrome major version (`v151`) and passes `version_main` to `undetected-chromedriver`. |
| **Empty Time Windows (No Tweets)** | `SignalGenerator` catches empty windows and outputs a neutral `HOLD` signal with `confidence=0.0` and zero variance without crashing. |
| **Single-Tweet Time Window ($N=1$)** | Variance calculation handles zero degrees of freedom gracefully by setting $\text{SEM}_w = 0$ and defaulting to `HOLD`. |
| **Devanagari / Emoji Unicode Crashes** | All file streams explicitly enforce `encoding="utf-8"`, and Windows console prints sanitize emoji characters to prevent `charmap` codec exceptions. |
| **Missing Market Data Candles** | `MarketValidator` skips time windows where corresponding OHLCV candles do not exist and computes accuracy strictly over matched intervals. |

---

### Summary
This architecture provides a **production-ready, data-driven, and statistically defensible** quantitative tweet intelligence system built specifically for the nuances of the Indian financial market.

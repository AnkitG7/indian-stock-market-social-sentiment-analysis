# Technical Approach

## 1. Architecture Overview

The system follows a layered pipeline architecture with clear separation of concerns:

```
Collection → Processing → Analysis → Validation → Reporting
```

Each layer communicates through shared domain models (`models.py`), ensuring no coupling between external dependencies and internal logic. The key architectural decision is the **Collector abstraction** — an `ABC` interface (`TweetCollector`) that decouples the unstable external scraping mechanism from the processing and analytics layers. If Twitter changes its internal APIs tomorrow, only the collector implementation changes; the rest of the pipeline remains untouched.

## 2. Data Collection Strategy

### Why Twikit + Selenium (not just one)

Twitter/X has made scraping increasingly difficult since 2023:
- Guest/unauthenticated access was removed entirely
- `snscrape` and `ntscraper` are both defunct
- The official API (v2) requires paid access

**Twikit** interacts directly with Twitter's internal GraphQL endpoints using authenticated session cookies. It's fast (no browser overhead), async, and provides structured tweet objects. However, it's fragile — Twitter periodically changes endpoint signatures.

**Selenium** (via `undetected-chromedriver`) is the fallback. It renders the full page in a real browser, making it resistant to API changes but expensive in RAM/CPU. We use stable `data-testid` selectors rather than obfuscated CSS classes.

The `TweetCollector` ABC ensures the pipeline only depends on the interface:

```python
class TweetCollector(ABC):
    @abstractmethod
    async def collect(self, query, since, until, limit) -> list[RawTweet]: ...
    
    @abstractmethod
    async def close(self) -> None: ...
```

### Rate Limiting

The `RateLimiter` implements:
- **Exponential backoff with jitter** on failure/429 responses
- Configurable `base_delay` (3s) and `max_delay` (60s)
- Reset on success, escalation on consecutive failures
- This mimics human browsing patterns and avoids automated IP bans

### Query Construction

The `QueryBuilder` constructs optimized Twitter advanced search queries:
```
(#nifty50 OR #banknifty OR #sensex OR #intraday) -is:retweet min_faves:2 lang:en
```
- Hashtag combination with OR logic
- Retweet exclusion to reduce noise
- Minimum engagement filter to exclude bot spam
- Date windowing with `since:` / `until:` for 24-hour collection

### Honest Collection Reporting

We never fabricate collection numbers. `CollectionStats` is a first-class object:
```
Requested tweets : 2000
Collected tweets : 1913
Duplicates       :   87
Final tweets     : 1826
Collection time  : 41m 12s
```

## 3. Text Processing

### Unicode and Hinglish Handling

Indian financial Twitter has three content types:
1. **English**: "Nifty broke 22,500 resistance with heavy call unwinding"
2. **Hindi (Devanagari)**: "बाजार में भारी बिकवाली"
3. **Hinglish (Code-mixed)**: "Bhai Nifty gap up hoga kal, call le lo"

Language detection uses a simple but effective heuristic:
- If >30% of characters are Devanagari (U+0900–U+097F) → `hi`
- If text contains 2+ common Hindi words in Roman script → `hinglish`
- Otherwise → `en`

### Emoji-to-Token Mapping

Financial tweets use emojis as sentiment signals:
- 🚀🐂📈🟢🔥💰✅ → `EMOJI_BULL`
- 🐻🩸📉🔴💣❌⚠️ → `EMOJI_BEAR`

These are converted to explicit tokens so the TF-IDF vectorizer and sentiment analyzer can process them.

### Ticker and Strike Normalization

- `$NIFTY`, `#nifty50`, `NIFTY`, `#BankNifty` → `TICKER_NIFTY`, `TICKER_BANKNIFTY`
- `22500CE`, `48500 PE` → `STRIKE_CE`, `STRIKE_PE`

### Financial Tokenizer

The tokenizer preserves terms that generic NLP pipelines would discard as stopwords:
```
up, down, call, put, long, short, buy, sell, bull, bear, breakout, breakdown, not, no
```
These carry 100% of trading sentiment and must never be removed.

Hinglish slang is canonicalized: `teji` → `bullish`, `mandi` → `bearish`, `SL hit` → `stoploss_hit`.

## 4. Deduplication

Indian financial Twitter is heavily polluted with spam bots, copy-paste advisory messages, and near-duplicate promotional content. We use multi-stage deduplication:

1. **Exact ID check** — `O(1)` set lookup
2. **Content hash** — Canonicalize text (strip URLs, mentions, punctuation), compute MD5 hash
3. **MinHash LSH** — For near-duplicates (bots adding random suffixes). Uses 128 permutations with Jaccard threshold ≥ 0.80 over 3-character shingles

All dedup state is bounded by a sliding window (`OrderedDict` with LRU eviction) to prevent unbounded memory growth.

## 5. Sentiment Analysis

We use VADER (Valence Aware Dictionary and sEntiment Reasoner) augmented with a custom Indian Financial Market Lexicon of 50+ terms:

**Bullish**: `teji` (+2.5), `breakout` (+2.2), `jackpot` (+2.8), `multibagger` (+2.8), `gap_up` (+1.8), `EMOJI_BULL` (+2.5)

**Bearish**: `mandi` (-2.5), `crash` (-3.2), `trap` (-2.4), `phans_gaye` (-2.5), `gap_down` (-1.8), `EMOJI_BEAR` (-2.5)

**Why VADER over transformers**: The assignment emphasizes production readiness and efficiency. VADER processes tweets in <0.1ms (vs 20-50ms for FinBERT), requires no GPU, and with the custom lexicon handles Indian market terminology well. The assignment explicitly permits TF-IDF, embeddings, or custom feature engineering.

## 6. TF-IDF Feature Extraction

Configuration:
- `ngram_range=(1, 3)` — captures multi-word directional phrases like "breakout above resistance" or "call writer trap"
- `sublinear_tf=True` — replaces raw TF with 1 + log(TF), preventing keyword-stuffing spam from dominating
- `min_df=3` — ignores single-tweet typos
- `max_df=0.85` — drops platform-wide boilerplate

## 7. Signal Generation

### Engagement-Weighted Sentiment

Not all tweets are equal. A tweet from a verified analyst with 500 likes carries more signal than a bot with 0 engagement:

$$w_i = \ln(1 + \text{likes}_i + 2 \cdot \text{retweets}_i + 0.5 \cdot \text{replies}_i) \cdot \ln(1 + \text{followers}_i)$$

### Exponential Time Decay

Information on financial Twitter decays rapidly. We apply exponential decay with a configurable half-life (default: 60 minutes):

$$\omega_i(t) = w_i \cdot \exp\left(-\frac{\ln 2}{t_{\text{half}}} \cdot (t - t_i)\right)$$

### Composite Score

The weighted sentiment mean over a time window:

$$\mu_t = \frac{\sum_{i} \omega_i \cdot S_i}{\sum_{i} \omega_i}$$

Combined with supplementary features:
- **Bull-Bear Ratio**: $(N_{\text{bull}} - N_{\text{bear}}) / (N_{\text{bull}} + N_{\text{bear}} + \epsilon)$
- **Sentiment Velocity**: $\mu_t - \mu_{t-1}$ (momentum)
- **Volume Anomaly Ratio**: window volume / rolling mean volume

### 95% Confidence Intervals

Using Weighted Standard Error of the Mean:

$$N_{\text{eff}} = \frac{(\sum \omega_i)^2}{\sum \omega_i^2}$$

$$s_w = \sqrt{\frac{\sum \omega_i (S_i - \mu_t)^2}{\sum \omega_i}}$$

$$\text{CI}_{0.95} = \mu_t \pm 1.96 \cdot \frac{s_w}{\sqrt{N_{\text{eff}}}}$$

### Signal Gating

Signals are only generated when confidence bounds clear conviction thresholds:
- **BUY**: CI lower bound > +0.2 AND volume anomaly > 1.5
- **SELL**: CI upper bound < -0.2 AND volume anomaly > 1.5
- **HOLD**: Otherwise (high uncertainty / low consensus)

## 8. Market Validation

The most important addition. We don't just generate sentiment — we validate it against actual price movements:

1. For each BUY/SELL signal, find NIFTY/BANKNIFTY price at signal time
2. Look ahead 15 minutes, find exit price
3. Calculate forward return
4. BUY is correct if return > 0; SELL is correct if return < 0

Metrics calculated:
- **BUY precision**: correct BUY predictions / total BUY signals
- **SELL precision**: correct SELL predictions / total SELL signals
- **Overall accuracy**: total correct / total signals
- **Signal coverage**: actionable signals / total time windows
- **Average forward return**: mean return following signal

The `MarketDataProvider` is also an ABC with `CSVMarketDataProvider` implementation, keeping the validation layer pluggable.

## 9. Scalability Considerations (10x Data)

| Component | Current Design | 10x Scaling Path |
|---|---|---|
| Scraping | Single account, sequential pagination | Account pool rotation (twscrape), parallel queries |
| Deduplication | In-memory OrderedDict with sliding window | Redis-backed Bloom filter + MinHash LSH index |
| Storage | Single Parquet file per run | Date-partitioned Parquet with row-group batching |
| TF-IDF | In-memory `TfidfVectorizer` | `HashingVectorizer` for streaming/incremental |
| Signal Gen | Batch over all tweets | Streaming window aggregation with `collections.deque` |
| Visualization | Full dataset rendering | LTTB downsampling (already implemented), Datashader for >1M points |

## 10. Design Decisions and Trade-offs

| Decision | Rationale |
|---|---|
| Parquet over SQLite/Postgres | No database server dependency, columnar compression (5-10x vs JSON), predicate pushdown for fast queries |
| VADER over FinBERT | CPU-only, <0.1ms/tweet, no GPU dependency, custom lexicon handles Indian terms well |
| ABC interfaces for collectors and market data | Isolates fragile external dependencies, enables testing with mocks, future-proof |
| Sliding window dedup instead of unbounded cache | Bounds memory at O(window_size), suitable for continuous streaming |
| Weighted SEM over Bootstrap CI | Analytical solution is faster than 1000 bootstrap iterations, sufficient for most distributions |
| `config.py` dataclasses over YAML/JSON | Type-safe, IDE-friendly, no external config parser needed |

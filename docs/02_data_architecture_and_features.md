# Data Ingestion, Caching, and Feature Engineering

## Data Sources
- `yfinance`: NASDAQ and commodity data. **Always use split/dividend-adjusted prices** (`auto_adjust=True`) for features; store both raw and adjusted if display needs raw.
- `tvDatafeed`: BIST (Turkey) market data. Note `tvDatafeed` is unofficial/scraping-based and breaks periodically — wrap it behind a source interface with a fallback and retry/backoff.

### Data hygiene (was missing — these are the bugs that quietly ruin a financial model)
- **Timezones:** store everything in **UTC**. NASDAQ candles are `America/New_York`, BIST is `Europe/Istanbul`. Mixing them corrupts intraday alignment and session boundaries.
- **De-duplication & gaps:** enforce unique `(asset_id, timeframe, timestamp)`; forward-fill is *not* allowed across non-trading sessions.
- **Indicator warm-up:** drop the leading rows where rolling indicators are still `NaN`. Never impute them — that injects look-ahead.
- **Survivorship bias:** be aware tickers that delisted won't appear; document this limitation.

---

## Storage & Caching Strategy

**Primary Database:** SQLite for persistent time-series storage.
- Use a **composite PRIMARY KEY `(asset_id, timeframe, timestamp)`** (not just an index) so ingestion is an idempotent UPSERT (`INSERT ... ON CONFLICT ... DO UPDATE`).
- Enable **WAL mode** (`PRAGMA journal_mode=WAL`) — without it, concurrent reads block on writes, which will stall the UI under multiple users.
- **HF caveat:** Spaces disk is **ephemeral (50GB, wiped on rebuild)**. The SQLite file must either be seeded at startup from an HF Dataset repo, or replaced by an external DB (Turso/libSQL, Supabase) in production. See deployment doc.

**Cache layer — abstract it, do NOT hard-code Redis.**
The previous spec mandated Redis. **Redis is not available on the HF free tier**, so a Redis-only design cannot deploy. Define a `CacheBackend` interface with two implementations:
- **Local dev:** Redis (via Docker/daemon) — fast, shared, supports TTL.
- **HF / fallback:** in-process TTL cache (`cachetools.TTLCache`) or `diskcache`, or an external managed Redis (e.g. Upstash free tier) injected via secret.

**Lookup order (unchanged, now backend-agnostic):** check Cache → on miss query SQLite → on miss call the API, then write back to SQLite and Cache. TTLs should be short for volatile intraday data (5m/15m BIST) and long for daily+.

---

## Feature Engineering (`TA-Lib` & `pandas-ta`)
- Core features: Moving Averages, RSI, MACD, Bollinger Bands, plus **ATR** (needed for volatility-scaled labeling below).
- Prefer C-based **TA-Lib** for speed. **Install caveat:** TA-Lib needs a compiled C library that is awkward on HF Spaces — pin the build or fall back to `pandas-ta` (pure-Python, slower, less actively maintained). Decide one canonical implementation per indicator to avoid train/serve skew.
- **All features computed on closed candles only.** Vectorize (NumPy/Pandas); no Python row loops.

---

## Target Labeling Strategy (Dealer Rules)

The original fixed-percentage table treats a 1% move in a mega-cap the same as in a volatile BIST small-cap — that is statistically wrong and produces noisy labels. Move to a **triple-barrier method** with **volatility-scaled barriers**.

For each sample, set three barriers over a forward horizon:
- **Upper (take-profit) / Lower (stop)** barriers scaled by recent volatility: `barrier = k * ATR` (or `k * rolling_std`), where `k` is tuned per horizon.
- **Vertical barrier:** a maximum holding period (the horizon). If neither price barrier is hit first, label = **Hold**.

The fixed percentages below are retained **only as default `k`-equivalents / sanity bounds**, not as hard rules:

| Horizon        | BUY (upper)        | SELL (lower)        | Vertical barrier |
|----------------|--------------------|---------------------|------------------|
| Short (5m–1H)  | +0.5% … +1.0%      | −1.0% … −2.5%       | end of horizon   |
| Medium (1D–1W) | +5.0% … +10.0%     | −5.0% … −10.0%      | end of horizon   |
| Long (1M+)     | +12.0% … +15.0%    | −12.0% … −15.0%     | end of horizon   |

**Leakage rules (critical):**
- The label looks *forward*; features look *backward*. The forward window used for labeling must be **purged/embargoed** from the training features of nearby samples (see CV section in the MLOps doc).
- Expect heavy class imbalance toward `Hold`; handle it at the model level, not by rebalancing away the real distribution blindly.

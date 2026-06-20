# Project Scope and UI Architecture

**Overview:** A web-based stock and commodity prediction application deployed initially on Hugging Face Spaces. It combines technical analysis with AI models (LightGBM, Attention-LSTM) to output **unified Buy / Hold / Sell** signals across multiple timeframes (5m to ~4 months).

> **Scope note:** The class set is **three-class (Buy / Hold / Sell)**, not binary. `Hold` is a first-class label and typically the majority class — this drives the imbalance handling described in `03_ai_models_and_mlops.md`. Any "Buy/Sell only" wording elsewhere is deprecated.

---

## UI Layout (Hugging Face Interface)

- **Left Panel:** Market selection (BIST, NASDAQ, Commodities), asset ticker search, and timeframe selector (5m / 15m / 1H / 4H / 1D / 1W / 1M).
- **Main Display:** Interactive line/candlestick chart of historical price action overlaid with the final Short / Medium / Long-term recommendation, the model confidence score, and the entry/target/stop levels implied by the active timeframe's labeling rule.
- **Dual-Tab Analysis:**
  - *Tab 1 — Indicators:* RSI, MACD, Bollinger Bands. Users adjust the **weight/influence** of each indicator via sliders. Weights are normalized to sum to 1 before aggregation.
  - *Tab 2 — AI Engine:* LightGBM and LSTM class probabilities. Users adjust the **confidence threshold** and the **relative weight** of each model.
- **Backtest / Track-record Panel (REQUIRED, was missing):** Displays out-of-sample performance of the currently selected configuration — hit rate, equity curve, max drawdown, and signal count — computed on a held-out walk-forward window. Without this, signals are unfalsifiable and the UI is misleading.
- **Disclaimer (REQUIRED):** A persistent, non-dismissable footer stating this is not financial advice and that past performance does not guarantee future results. This is both an ethical and a legal requirement for a tool emitting Buy/Sell signals.

---

## Signal Aggregation Engine (the core that was undefined)

The previous spec said signals are "unified" but never defined *how*. This is the most important architectural decision in the project, so it is specified here.

Each source produces a **probability vector** over `{Buy, Hold, Sell}`:

1. **Indicator signals** → rule-based functions map each indicator to a soft vote (e.g., RSI < 30 leans Buy). User indicator-weights `w_ind` scale these.
2. **LightGBM** → calibrated class probabilities. User weight `w_lgbm`.
3. **LSTM** → calibrated class probabilities. User weight `w_lstm`.

**Fusion rule (soft voting):**
```
P_final = (w_ind * P_indicators + w_lgbm * P_lgbm + w_lstm * P_lstm) / (w_ind + w_lgbm + w_lstm)
label   = argmax(P_final)  if max(P_final) >= user_confidence_threshold  else "Hold"
confidence = max(P_final)
```

- Probabilities **must be calibrated** (see MLOps doc) or the threshold slider is meaningless.
- The engine returns a single standardized result object, e.g.
  `{ "label", "confidence", "per_source_probs", "timeframe", "levels": {entry, target, stop} }`,
  consumed identically by the UI and any future API.

---

## Execution Flow (DRY)

Timeframe is passed dynamically to a **single Analysis Engine** (`inference/analysis_engine.py`). The same code path serves every timeframe and both UI tabs — no per-timeframe branching of business logic. The engine:

1. Resolves the cache → DB → API data fetch (see data doc).
2. Computes features on **closed candles only** (no look-ahead).
3. Runs inference (ONNX/quantized on HF).
4. Calls the Signal Aggregation Engine.
5. Returns the standardized result object.

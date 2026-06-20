---
title: Borsacım — AI Market Signals
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Borsacım

A stock/commodity Buy/Hold/Sell signal app for BIST, NASDAQ, and commodities. Combines rule-based technical indicators (RSI, MACD, Bollinger Bands) with calibrated LightGBM and Attention-LSTM model probabilities via a soft-voting Signal Aggregation Engine, with a required out-of-sample backtest panel for transparency.

**Not financial advice.** See the in-app disclaimer.

## Architecture

See `docs/01_project_scope_and_ui.md` through `docs/04_deployment_and_environment.md` for the full design spec. Summary:

- `data_pipeline/` — yfinance (NASDAQ/commodities) + tvDatafeed (BIST) ingestion, vectorized feature engineering, triple-barrier labeling.
- `ml_pipeline/` — purged/embargoed walk-forward CV, LightGBM + Attention-LSTM training (Optuna + MLflow), probability calibration, ONNX export. **Local-only — never runs on this Space.**
- `inference/` — loads ONNX champions + calibrators, runs the fusion rule, the single DRY analysis engine.
- `backtest/` — replays the live fusion rule over history for the UI's track-record panel.
- `app.py` — the Streamlit UI running on this Space.

## Models

This Space only performs inference. Trained model artifacts (ONNX + scaler + calibrators) are pulled at runtime from the `HF_DATASET_REPO` configured as a Space variable — they are never committed to this repo (docs/04). If no champion exists yet for a given (symbol, timeframe), the app degrades gracefully to indicators-only signals.

## Local development

```bash
uv sync
uv run streamlit run app.py
```

Training (local only, needs the heavier `requirements-train.txt` stack):

```bash
uv run python -c "from ml_pipeline.train_lightgbm import run_training; run_training(symbol='AAPL', timeframe='1D')"
```

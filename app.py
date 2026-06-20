"""HF Spaces main UI (docs/01). Streamlit only — this file never trains;
it only calls inference.analysis_engine / backtest.walk_forward, which
in turn only load artifacts ml_pipeline/export_onnx.py already produced
(docs/03's train/serve boundary).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest.walk_forward import run_backtest
from data_pipeline.feature_engine import compute_features, drop_warmup
from inference.analysis_engine import analyze, drop_unclosed_candle, fetch_ohlcv_cached
from inference.model_loader import try_load_model_bundle
from ml_pipeline.common import FEATURE_COLUMNS

st.set_page_config(page_title="Borsacım — AI Market Signals", layout="wide")

MARKETS = {
    "NASDAQ": {"source_kind": "yfinance", "example": "AAPL"},
    "Commodity": {"source_kind": "yfinance", "example": "GC=F"},
    "BIST": {"source_kind": "tvdatafeed", "example": "THYAO"},
}
TIMEFRAMES = ["5m", "15m", "1H", "4H", "1D", "1W", "1M"]

# Short(5m-1H), Medium(1D-1W), Long(1M+) — see docs/02's labeling table.
TIMEFRAME_TO_HORIZON_BUCKET = {
    "5m": "short", "15m": "short", "1H": "short", "4H": "short",
    "1D": "medium", "1W": "medium", "1M": "long",
}
# Default forward horizon (bars) for the triple-barrier rule per
# timeframe — approximate; a production deployment would store the
# exact horizon_bars each champion was actually trained with (e.g. in
# meta.json) rather than re-guess it here.
TIMEFRAME_TO_HORIZON_BARS = {
    "5m": 12, "15m": 8, "1H": 6, "4H": 6, "1D": 10, "1W": 8, "1M": 6,
}

MODELS_DIR = Path(__file__).parent / "ml_pipeline" / "models"

DISCLAIMER = (
    "⚠️ **Not financial advice.** This tool produces probabilistic, model-generated signals for "
    "educational/research purposes only. Past performance (including the backtest panel below) "
    "does not guarantee future results. Markets carry risk of loss — always do your own research "
    "and consult a licensed advisor before trading."
)


@st.cache_resource(show_spinner=False)
def _load_bundles(symbol: str, timeframe: str):
    # path_in_repo mirrors the local layout (ml_pipeline/models/{symbol}_
    # {timeframe}/{model_type}) inside the configured HF dataset repo —
    # without it, the HF Hub fallback in ensure_local_artifacts() would
    # download the whole repo and look for model.onnx at its root instead
    # of the right subfolder.
    repo_subdir = f"{symbol}_{timeframe}"
    model_dir = MODELS_DIR / repo_subdir
    lgbm_bundle = try_load_model_bundle("lightgbm", model_dir / "lightgbm", path_in_repo=f"{repo_subdir}/lightgbm")
    lstm_bundle = try_load_model_bundle("lstm", model_dir / "lstm", path_in_repo=f"{repo_subdir}/lstm")
    return lgbm_bundle, lstm_bundle


def _candlestick_chart(df: pd.DataFrame, levels: dict) -> go.Figure:
    fig = go.Figure(
        data=[go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price")]
    )
    if levels.get("target") is not None:
        fig.add_hline(y=levels["target"], line_dash="dot", line_color="green", annotation_text="Target")
    if levels.get("stop") is not None:
        fig.add_hline(y=levels["stop"], line_dash="dot", line_color="red", annotation_text="Stop")
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
    return fig


def main() -> None:
    st.title("Borsacım — AI Market Signals")

    with st.sidebar:
        st.header("Market & Asset")
        market = st.selectbox("Market", list(MARKETS.keys()))
        # Ticker entry is form-gated so fetches don't fire on every
        # keystroke while typing — only on submit.
        with st.form("ticker_form"):
            symbol_input = st.text_input("Ticker", value=MARKETS[market]["example"])
            submitted = st.form_submit_button("Load", type="primary")
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1D"))

    if "symbol" not in st.session_state or submitted:
        st.session_state.symbol = symbol_input.strip().upper()
    symbol = st.session_state.symbol

    horizon_bucket = TIMEFRAME_TO_HORIZON_BUCKET[timeframe]
    horizon_bars = TIMEFRAME_TO_HORIZON_BARS[timeframe]

    tab_indicators, tab_ai = st.tabs(["Indicators", "AI Engine"])
    with tab_indicators:
        st.caption("Per-indicator weight/influence — normalized to sum to 1 before aggregation.")
        rsi_w = st.slider("RSI weight", 0.0, 1.0, 1.0)
        macd_w = st.slider("MACD weight", 0.0, 1.0, 1.0)
        bb_w = st.slider("Bollinger Bands weight", 0.0, 1.0, 1.0)

    with tab_ai:
        st.caption("Confidence threshold below which the fused signal falls back to Hold.")
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.4)
        w_ind = st.slider("Indicators weight", 0.0, 1.0, 1.0)
        w_lgbm = st.slider("LightGBM weight", 0.0, 1.0, 1.0)
        w_lstm = st.slider("LSTM weight", 0.0, 1.0, 1.0)

    indicator_weights = {"rsi": rsi_w, "macd": macd_w, "bollinger": bb_w}

    if not symbol:
        st.info("Enter a ticker in the sidebar and click Analyze.")
        st.markdown("---")
        st.warning(DISCLAIMER)
        return

    lgbm_bundle, lstm_bundle = _load_bundles(symbol, timeframe)
    if lgbm_bundle is None and lstm_bundle is None:
        st.info(
            f"No trained champion found for {symbol} ({timeframe}) — falling back to indicators-only fusion. "
            f"Train one via ml_pipeline/train_lightgbm.py + train_lstm.py, then export with "
            f"ml_pipeline/export_onnx.py into ml_pipeline/models/{symbol}_{timeframe}/."
        )

    try:
        with st.spinner("Fetching data and computing signal..."):
            df = fetch_ohlcv_cached(symbol, market, timeframe, lookback_days=365)
            df = drop_unclosed_candle(df, timeframe)
            if df.empty:
                st.error(f"No data available for {symbol} ({timeframe}).")
                st.markdown("---")
                st.warning(DISCLAIMER)
                return

            result = analyze(
                symbol=symbol, market=market, timeframe=timeframe,
                indicator_weights=indicator_weights,
                w_ind=w_ind, w_lgbm=w_lgbm, w_lstm=w_lstm,
                confidence_threshold=confidence_threshold,
                horizon_bucket=horizon_bucket,
                lgbm_bundle=lgbm_bundle, lstm_bundle=lstm_bundle,
                ohlcv_df=df,
            )
    except ValueError as exc:
        st.error(str(exc))
        st.markdown("---")
        st.warning(DISCLAIMER)
        return

    label_color = {"Buy": "green", "Sell": "red", "Hold": "gray"}[result.label]
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"### :{label_color}[{result.label}]")
    col2.metric(f"Confidence ({horizon_bucket}-term)", f"{result.confidence:.0%}")
    if result.levels.get("target") is not None:
        col3.metric("Entry / Target / Stop", f"{result.levels['entry']:.2f} / {result.levels['target']:.2f} / {result.levels['stop']:.2f}")
    else:
        col3.metric("Entry", f"{result.levels['entry']:.2f}")

    st.plotly_chart(_candlestick_chart(df, result.levels), use_container_width=True)

    with tab_ai:
        st.subheader("Per-source class probabilities")
        prob_df = pd.DataFrame(
            {k: v for k, v in result.per_source_probs.items() if v is not None},
            index=["Sell", "Hold", "Buy"],
        )
        st.bar_chart(prob_df)

    st.subheader("Backtest — out-of-sample track record")
    st.caption(
        "Hit rate, equity curve, and drawdown of THIS exact configuration (current weights/threshold), "
        "replayed over historical closed bars with the same triple-barrier exit rule used in training."
    )
    if st.button("Run backtest"):
        with st.spinner("Running walk-forward backtest..."):
            report = run_backtest(
                df, feature_columns=FEATURE_COLUMNS, horizon_bars=horizon_bars, horizon_bucket=horizon_bucket,
                indicator_weights=indicator_weights, w_ind=w_ind, w_lgbm=w_lgbm, w_lstm=w_lstm,
                confidence_threshold=confidence_threshold, lgbm_bundle=lgbm_bundle, lstm_bundle=lstm_bundle,
            )
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        bcol1.metric("Hit rate", f"{report.hit_rate:.0%}" if report.hit_rate == report.hit_rate else "n/a")
        bcol2.metric("Net PnL", f"{report.net_pnl:.2%}")
        bcol3.metric("Max drawdown", f"{report.max_drawdown:.2%}")
        bcol4.metric("Signal count", report.n_signals)
        if report.equity_curve:
            st.line_chart(pd.Series(report.equity_curve, name="Cumulative return"))
        else:
            st.info("No trades were triggered by this configuration over the available history.")

    st.markdown("---")
    st.warning(DISCLAIMER)


if __name__ == "__main__":
    main()

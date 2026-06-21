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
from core.db_setup import init_db
from data_pipeline.sources.base import DataSourceError
from inference.analysis_engine import analyze, drop_unclosed_candle, fetch_ohlcv_cached
from inference.model_loader import try_load_model_bundle
from ml_pipeline.common import FEATURE_COLUMNS
from utils.logging_config import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Borsacım — AI Market Signals", layout="wide")


@st.cache_resource(show_spinner=False)
def _init_db_once() -> None:
    # HF Spaces' disk is ephemeral (docs/04) — the SQLite schema must be
    # (re)created on every fresh container, not just once at dev-machine
    # setup time. st.cache_resource ensures this runs once per process,
    # not on every script rerun.
    init_db()


_init_db_once()

MARKETS = {
    "NASDAQ": {"source_kind": "yfinance", "example": "AAPL"},
    "Commodity": {"source_kind": "yfinance", "example": "GC=F"},
    "BIST": {"source_kind": "tvdatafeed", "example": "THYAO"},
}
# Investment-horizon framing (per user request, replacing the previous
# candle-granularity-driven short/medium/long split): the dealer-relevant
# question is "how much history informs this decision and how long is
# the implied holding period", not "which candle size". Each bucket
# fixes the candle granularity and lookback window together — an
# intraday candle size doesn't make sense paired with a 10-year lookback
# (millions of bars), and a monthly candle doesn't make sense for a
# 1-year lookback (~12 bars, not enough for indicator warm-up).
HORIZON_OPTIONS = {
    "short": "Short-term (1D candles, up to 1 year)",
    "medium": "Medium-term (1D candles, up to 5 years)",
    "long": "Long-term (1W candles, up to 10 years)",
}
HORIZON_TO_TIMEFRAME = {"short": "1D", "medium": "1D", "long": "1W"}
HORIZON_TO_LOOKBACK_DAYS = {"short": 365, "medium": 1825, "long": 3650}
# Forward horizon (bars) for the triple-barrier rule, scaled with the
# holding period a horizon bucket implies: short ~ a couple of weeks of
# daily bars, medium ~ a month and a half, long ~ a quarter of weekly
# bars. Approximate by design — a production deployment would store the
# exact horizon_bars each champion was actually trained with (e.g. in
# meta.json) rather than re-derive it here.
HORIZON_TO_HORIZON_BARS = {"short": 10, "medium": 30, "long": 13}

# "Short-term" additionally exposes a candle-granularity sub-choice (per
# user request) since "short-term trading" spans very different bar
# sizes in practice (intraday scalping vs. swing trading on daily bars).
# Each granularity gets its OWN lookback, not the "short-term" bucket's
# 365-day default: 1H/6H/12H are capped by yfinance/tvDatafeed's ~2-year
# intraday history limit, while 1W/1M need a multi-year lookback
# regardless of the "short-term" label just to clear indicator warm-up
# (close_sma50_ratio needs >=50 bars — 50 *weeks* is ~1 year, 50
# *months* is ~4 years; a literal "up to 1 year" lookback for the 1M
# choice would silently fail with "not enough history").
SHORT_TERM_SUB_OPTIONS = {
    "1H": "1 Hour (up to ~2 years)",
    "6H": "6 Hours (up to ~2 years)",
    "12H": "12 Hours (up to ~2 years)",
    "1D": "1 Day (up to 1 year)",
    "1W": "1 Week (up to 3 years)",
    "1M": "1 Month (up to 10 years)",
}
SHORT_SUB_TO_LOOKBACK_DAYS = {"1H": 729, "6H": 729, "12H": 729, "1D": 365, "1W": 1095, "1M": 3650}
# Same "approximate by design" reasoning as HORIZON_TO_HORIZON_BARS,
# scaled down to a roughly multi-day-to-few-week holding period per bar size.
SHORT_SUB_TO_HORIZON_BARS = {"1H": 24, "6H": 8, "12H": 6, "1D": 10, "1W": 4, "1M": 3}

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
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price")])
    if levels.get("target") is not None:
        fig.add_hline(y=levels["target"], line_dash="dot", line_color="green", annotation_text="Target")
    if levels.get("stop") is not None:
        fig.add_hline(y=levels["stop"], line_dash="dot", line_color="red", annotation_text="Stop")
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
    return fig


_SOURCE_COLORS = {"indicators": "#7fb3ff", "lightgbm": "#ff6b6b", "lstm": "#51cf66", "final": "#1f2937"}


def _per_source_probability_chart(per_source_probs: dict) -> go.Figure:
    # Grouped (side-by-side), not stacked: these are independent
    # probability distributions from different sources, not parts of a
    # whole — stacking them (Streamlit's st.bar_chart default for
    # multi-column data) visually implies a meaningless combined total.
    classes = ["Sell", "Hold", "Buy"]
    fig = go.Figure()
    for name, probs in per_source_probs.items():
        if probs is None:
            continue
        fig.add_trace(go.Bar(name=name, x=classes, y=probs, marker_color=_SOURCE_COLORS.get(name)))
    fig.update_layout(
        barmode="group",
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        legend_title_text="Source",
    )
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

    if "symbol" not in st.session_state or submitted:
        st.session_state.symbol = symbol_input.strip().upper()
    symbol = st.session_state.symbol

    tab_indicators, tab_ai = st.tabs(["Indicators", "AI Engine"])
    with tab_indicators:
        horizon_bucket = st.selectbox("Investment Horizon", list(HORIZON_OPTIONS.keys()), index=1, format_func=lambda k: HORIZON_OPTIONS[k])
        if horizon_bucket == "short":
            sub_tf = st.selectbox(
                "Short-term timeframe",
                list(SHORT_TERM_SUB_OPTIONS.keys()),
                index=3,  # "1D" — same default as before this sub-choice existed
                format_func=lambda k: SHORT_TERM_SUB_OPTIONS[k],
            )
            timeframe = sub_tf
            lookback_days = SHORT_SUB_TO_LOOKBACK_DAYS[sub_tf]
            horizon_bars = SHORT_SUB_TO_HORIZON_BARS[sub_tf]
        else:
            timeframe = HORIZON_TO_TIMEFRAME[horizon_bucket]
            lookback_days = HORIZON_TO_LOOKBACK_DAYS[horizon_bucket]
            horizon_bars = HORIZON_TO_HORIZON_BARS[horizon_bucket]

        st.divider()
        st.caption(
            "Per-indicator weight/influence — normalized to sum to 1, blended into one 'Indicators' signal "
            "that itself becomes one of three inputs to the final fusion (see the AI Engine tab)."
        )
        rsi_w = st.slider("RSI weight", 0.0, 1.0, 1.0)
        macd_w = st.slider("MACD weight", 0.0, 1.0, 1.0)
        bb_w = st.slider("Bollinger Bands weight", 0.0, 1.0, 1.0)

    with tab_ai:
        st.caption(
            "Confidence threshold below which the fused signal falls back to Hold. "
            "Random-guess baseline for 3 classes is ≈33% — set meaningfully above that."
        )
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)
        st.caption(
            "These three weights combine the *blended* Indicators signal (Tab 1 — already a weighted mix "
            "of RSI/MACD/Bollinger, not each indicator individually) with the LightGBM and LSTM model "
            "outputs into one final probability."
        )
        w_ind = st.slider("Technical Indicators (blended) weight", 0.0, 1.0, 1.0)
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
            df = fetch_ohlcv_cached(symbol, market, timeframe, lookback_days=lookback_days)
            df = drop_unclosed_candle(df, timeframe)
            if df.empty:
                st.error(f"No data available for {symbol} ({timeframe}).")
                st.markdown("---")
                st.warning(DISCLAIMER)
                return

            result = analyze(
                symbol=symbol,
                market=market,
                timeframe=timeframe,
                indicator_weights=indicator_weights,
                w_ind=w_ind,
                w_lgbm=w_lgbm,
                w_lstm=w_lstm,
                confidence_threshold=confidence_threshold,
                horizon_bucket=horizon_bucket,
                lgbm_bundle=lgbm_bundle,
                lstm_bundle=lstm_bundle,
                ohlcv_df=df,
            )
    except (ValueError, DataSourceError) as exc:
        st.error(f"Couldn't analyze {symbol}: {exc}")
        st.markdown("---")
        st.warning(DISCLAIMER)
        return
    except Exception:
        # Anything unanticipated (a third-party library quirk, a flaky
        # upstream API, etc.) must still degrade to a clean message —
        # showing a Python traceback to an end user isn't acceptable for
        # a tool meant to be typed a ticker by anyone. The real
        # exception is still logged server-side for debugging.
        logger.exception("Unexpected error analyzing %s (%s, %s)", symbol, market, timeframe)
        st.error(f"Something went wrong analyzing {symbol}. Please try again or pick a different ticker.")
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

    with tab_indicators:
        st.subheader("Current readings")
        ind = result.indicator_readings or {}
        icol1, icol2, icol3 = st.columns(3)
        rsi_val = ind.get("rsi")
        if rsi_val is not None:
            rsi_state = "Oversold" if rsi_val < 35 else "Overbought" if rsi_val > 65 else "Neutral"
            icol1.metric("RSI (14)", f"{rsi_val:.1f}", rsi_state)
        macd_val = ind.get("macd_hist")
        if macd_val is not None:
            macd_state = "Bullish" if macd_val > 0 else "Bearish" if macd_val < 0 else "Neutral"
            icol2.metric("MACD histogram", f"{macd_val:.3f}", macd_state)
        bb_val = ind.get("bb_percent_b")
        if bb_val is not None:
            bb_state = "Near lower band" if bb_val < 0.3 else "Near upper band" if bb_val > 0.7 else "Mid-range"
            icol3.metric("Bollinger %B", f"{bb_val:.2f}", bb_state)

    with tab_ai:
        st.subheader("Per-source class probabilities")
        calibrated = [name for name, on in (result.calibrated_sources or {}).items() if on]
        if calibrated:
            st.caption(
                f"{' and '.join(s.upper() if s == 'lstm' else s.title() for s in calibrated)} probabilities "
                "shown are calibrated (isotonic/Platt scaling) — not the model's raw output."
            )
        st.plotly_chart(_per_source_probability_chart(result.per_source_probs), use_container_width=True)

    st.subheader("Backtest — out-of-sample track record")
    st.caption(
        "Hit rate, equity curve, and drawdown of THIS exact configuration (current weights/threshold), "
        "replayed over historical closed bars with the same triple-barrier exit rule used in training."
    )
    if st.button("Run backtest"):
        with st.spinner("Running walk-forward backtest..."):
            report = run_backtest(
                df,
                feature_columns=FEATURE_COLUMNS,
                horizon_bars=horizon_bars,
                horizon_bucket=horizon_bucket,
                indicator_weights=indicator_weights,
                w_ind=w_ind,
                w_lgbm=w_lgbm,
                w_lstm=w_lstm,
                confidence_threshold=confidence_threshold,
                lgbm_bundle=lgbm_bundle,
                lstm_bundle=lstm_bundle,
            )
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        bcol1.metric("Hit rate", f"{report.hit_rate:.0%}" if report.hit_rate == report.hit_rate else "n/a")
        bcol2.metric("Net PnL", f"{report.net_pnl:.2%}")
        bcol3.metric("Max drawdown", f"{report.max_drawdown:.2%}")
        bcol4.metric("Signal count", report.n_signals)
        if report.equity_curve:
            st.line_chart(pd.Series(report.equity_curve, name="Cumulative return"))

            st.caption(f"Most recent trades for {symbol} under this exact configuration (newest first):")
            trades_df = pd.DataFrame(
                [
                    {
                        "Entry date": t.entry_date.strftime("%Y-%m-%d"),
                        "Exit date": t.exit_date.strftime("%Y-%m-%d"),
                        "Direction": t.direction,
                        "Return": f"{t.return_pct:+.2%}",
                        "Result": "Win" if t.win else "Loss",
                    }
                    for t in reversed(report.trades)
                ]
            )
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades were triggered by this configuration over the available history.")

    st.markdown("---")
    st.warning(DISCLAIMER)


if __name__ == "__main__":
    main()

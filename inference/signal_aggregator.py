"""Signal Aggregation Engine (docs/01) — the core that was previously
undefined: how indicator + LightGBM + LSTM outputs become one
Buy/Hold/Sell label.

Each source produces a probability vector over [Sell, Hold, Buy]
(matching data_pipeline.labeling.LABEL_TO_INT's order). Indicator votes
are rule-based soft scores (not a model), scaled by user-adjustable
per-indicator weights (Tab 1 sliders) before being combined into
P_indicators; that combined vector is itself one of three sources fused
with the LightGBM and LSTM calibrated probabilities (Tab 2 sliders) via:

    P_final = (w_ind*P_indicators + w_lgbm*P_lgbm + w_lstm*P_lstm)
              / (w_ind + w_lgbm + w_lstm)
    label   = argmax(P_final) if max(P_final) >= confidence_threshold else "Hold"
    confidence = max(P_final)

Any source whose weight is 0 or whose probabilities are None (model
unavailable) drops out of the fusion entirely rather than contributing
zeros that would silently drag P_final down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_pipeline.labeling import HORIZON_DEFAULTS, INT_TO_LABEL

# Indices into the [Sell, Hold, Buy] probability vectors, matching
# data_pipeline.labeling.LABEL_TO_INT.
SELL, HOLD, BUY = 0, 1, 2


@dataclass
class SignalResult:
    label: str
    confidence: float
    per_source_probs: dict
    timeframe: str
    levels: dict
    indicator_readings: dict | None = None  # raw {"rsi": .., "macd_hist": .., "bb_percent_b": ..} — for display, not fusion
    calibrated_sources: dict | None = None  # {"lightgbm": True, "lstm": False} — whether predict_proba applied calibration


def _normalize(raw: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw, 0.0, None)
    total = clipped.sum()
    if total <= 0:
        return np.full_like(clipped, 1.0 / len(clipped))
    return clipped / total


def rsi_vote(rsi: float) -> np.ndarray:
    """RSI < 35 leans Buy (oversold), > 65 leans Sell (overbought)."""
    buy = np.clip((35.0 - rsi) / 35.0, 0.0, 1.0)
    sell = np.clip((rsi - 65.0) / 35.0, 0.0, 1.0)
    hold = max(0.05, 1.0 - abs(rsi - 50.0) / 50.0)
    return _normalize(np.array([sell, hold, buy]))


def macd_vote(macd_hist: float, atr: float) -> np.ndarray:
    """MACD histogram normalized by ATR (scale-invariant across
    assets/volatility regimes) and squashed to [-1, 1]; positive ->
    bullish (Buy), negative -> bearish (Sell)."""
    normalized = macd_hist / atr if atr and atr > 0 else 0.0
    squashed = float(np.tanh(normalized))
    buy = max(0.0, squashed)
    sell = max(0.0, -squashed)
    hold = max(0.05, 1.0 - abs(squashed))
    return _normalize(np.array([sell, hold, buy]))


def bollinger_vote(percent_b: float) -> np.ndarray:
    """%B < 0.3 leans Buy (near/below lower band), > 0.7 leans Sell
    (near/above upper band)."""
    buy = np.clip((0.3 - percent_b) / 0.5, 0.0, 1.0)
    sell = np.clip((percent_b - 0.7) / 0.5, 0.0, 1.0)
    hold = max(0.05, 1.0 - abs(percent_b - 0.5) / 0.5)
    return _normalize(np.array([sell, hold, buy]))


def combine_indicator_votes(
    rsi: float,
    macd_hist: float,
    atr: float,
    percent_b: float,
    indicator_weights: dict[str, float] | None = None,
) -> np.ndarray:
    """P_indicators: weighted average of the three rule-based votes.
    Weights are normalized to sum to 1 before aggregation (docs/01)."""
    weights = indicator_weights or {"rsi": 1.0, "macd": 1.0, "bollinger": 1.0}
    w = np.array([weights.get("rsi", 0.0), weights.get("macd", 0.0), weights.get("bollinger", 0.0)])
    w_sum = w.sum()
    w = w / w_sum if w_sum > 0 else np.full(3, 1 / 3)

    votes = np.stack([rsi_vote(rsi), macd_vote(macd_hist, atr), bollinger_vote(percent_b)])
    return w @ votes


def fuse_signals(
    P_indicators: np.ndarray | None,
    P_lgbm: np.ndarray | None,
    P_lstm: np.ndarray | None,
    w_ind: float,
    w_lgbm: float,
    w_lstm: float,
    confidence_threshold: float,
    timeframe: str,
    close: float,
    atr: float,
    horizon_bucket: str = "medium",
    indicator_readings: dict | None = None,
    calibrated_sources: dict | None = None,
) -> SignalResult:
    sources = [(P_indicators, w_ind, "indicators"), (P_lgbm, w_lgbm, "lightgbm"), (P_lstm, w_lstm, "lstm")]
    active = [(p, w) for p, w, _ in sources if p is not None and w > 0]
    if not active:
        raise ValueError("At least one source with positive weight and non-None probabilities is required")

    weight_sum = sum(w for _, w in active)
    p_final = sum(w * p for p, w in active) / weight_sum

    confidence = float(p_final.max())
    label = INT_TO_LABEL[int(p_final.argmax())] if confidence >= confidence_threshold else "Hold"

    per_source_probs = {name: (p.tolist() if p is not None else None) for p, _, name in sources}
    per_source_probs["final"] = p_final.tolist()

    levels = compute_levels(label, close, atr, horizon_bucket)

    return SignalResult(
        label=label,
        confidence=confidence,
        per_source_probs=per_source_probs,
        timeframe=timeframe,
        levels=levels,
        indicator_readings=indicator_readings,
        calibrated_sources=calibrated_sources,
    )


def compute_levels(label: str, close: float, atr: float, horizon_bucket: str) -> dict:
    """Entry/target/stop implied by the same triple-barrier rule used to
    train the labels (docs/01) — never a different, inconsistent
    risk convention between training and the UI."""
    defaults = HORIZON_DEFAULTS[horizon_bucket]
    if label == "Buy":
        return {"entry": close, "target": close + defaults["k_upper"] * atr, "stop": close - defaults["k_lower"] * atr}
    if label == "Sell":
        return {"entry": close, "target": close - defaults["k_lower"] * atr, "stop": close + defaults["k_upper"] * atr}
    return {"entry": close, "target": None, "stop": None}

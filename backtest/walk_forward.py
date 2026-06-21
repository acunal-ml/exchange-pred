"""Out-of-sample track record for the UI's backtest panel (docs/01):

"Displays out-of-sample performance of the currently selected
configuration — hit rate, equity curve, max drawdown, and signal count
— computed on a held-out walk-forward window. Without this, signals
are unfalsifiable and the UI is misleading."

This replays the *exact* fused-signal decision rule (the user's current
indicator/model weights and confidence threshold, run through
inference.signal_aggregator.fuse_signals) over historical closed bars,
then scores each non-Hold call with the same triple-barrier exit
convention used to train the labels (data_pipeline.labeling) and the
same cost-aware PnL math used during training
(ml_pipeline.financial_metrics) — one realized-outcome definition,
reused everywhere, so the backtest panel can't quietly use a rosier
yardstick than training did.

Model inference here is batched (one ONNX call over the whole history),
not a per-bar Python loop calling the model — only the indicator votes
and the fusion step are per-bar, and those are pure arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_pipeline.feature_engine import compute_features, drop_warmup
from data_pipeline.labeling import HORIZON_DEFAULTS, LABEL_TO_INT, triple_barrier_labels
from inference.model_loader import ModelBundle, predict_proba
from inference.signal_aggregator import combine_indicator_votes, fuse_signals
from ml_pipeline.financial_metrics import DEFAULT_COST_BPS, compounded_report, sequential_trade_returns


@dataclass
class BacktestReport:
    hit_rate: float
    net_pnl: float
    sharpe: float
    max_drawdown: float
    n_signals: int
    equity_curve: list[float]
    label_counts: dict[str, int]


def _batched_lstm_proba(lstm_bundle: ModelBundle, features: np.ndarray) -> np.ndarray:
    """Probabilities for every position with a full seq_len history,
    NaN-padded for the warm-up positions that don't have one — one
    batched ONNX call, not a per-position loop."""
    seq_len = lstm_bundle.seq_len
    n, n_features = features.shape
    out = np.full((n, 3), np.nan)
    if n < seq_len:
        return out

    scaled = lstm_bundle.scaler.transform(features).astype(np.float32) if lstm_bundle.scaler else features.astype(np.float32)
    windows = np.stack([scaled[i - seq_len + 1 : i + 1] for i in range(seq_len - 1, n)])
    out[seq_len - 1 :] = predict_proba(lstm_bundle, windows)
    return out


def run_backtest(
    df: pd.DataFrame,
    feature_columns: list[str],
    horizon_bars: int,
    horizon_bucket: str = "medium",
    indicator_weights: dict[str, float] | None = None,
    w_ind: float = 1.0,
    w_lgbm: float = 1.0,
    w_lstm: float = 1.0,
    confidence_threshold: float = 0.4,
    lgbm_bundle: ModelBundle | None = None,
    lstm_bundle: ModelBundle | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
) -> BacktestReport:
    feats = drop_warmup(compute_features(df))
    if feats.empty:
        return BacktestReport(0.0, 0.0, float("nan"), 0.0, 0, [], {})

    defaults = HORIZON_DEFAULTS[horizon_bucket]
    barriers = triple_barrier_labels(feats, horizon_bars=horizon_bars, k_upper=defaults["k_upper"], k_lower=defaults["k_lower"])
    label_end_idx = barriers["label_end_idx"].to_numpy()
    close = feats["close"].to_numpy()
    n = len(feats)

    P_lgbm_all = predict_proba(lgbm_bundle, feats[feature_columns].to_numpy(dtype=np.float32)) if lgbm_bundle else None
    P_lstm_all = _batched_lstm_proba(lstm_bundle, feats[feature_columns].to_numpy(dtype=np.float32)) if lstm_bundle else None

    rsi = feats["rsi_14"].to_numpy()
    macd_hist = feats["macd_hist"].to_numpy()
    atr = feats["atr_14"].to_numpy()
    percent_b = feats["bb_percent_b"].to_numpy()

    label_ints = np.full(n, LABEL_TO_INT["Hold"], dtype=int)
    for i in range(n):
        if label_end_idx[i] < 0:
            continue  # no full forward window — can't be scored, skip the decision entirely

        p_ind = combine_indicator_votes(rsi[i], macd_hist[i], atr[i], percent_b[i], indicator_weights)
        p_lgbm = P_lgbm_all[i] if P_lgbm_all is not None else None
        p_lstm = P_lstm_all[i] if P_lstm_all is not None and not np.isnan(P_lstm_all[i]).any() else None

        result = fuse_signals(
            P_indicators=p_ind,
            P_lgbm=p_lgbm,
            P_lstm=p_lstm,
            w_ind=w_ind,
            w_lgbm=w_lgbm if p_lgbm is not None else 0.0,
            w_lstm=w_lstm if p_lstm is not None else 0.0,
            confidence_threshold=confidence_threshold,
            timeframe="",
            close=float(close[i]),
            atr=float(atr[i]),
            horizon_bucket=horizon_bucket,
        )
        label_ints[i] = LABEL_TO_INT[result.label]

    # Sequentially compounded, single-capital-pool returns — not the
    # additive sum of every signal's independent return, which would
    # overstate what one capital pool could realize once trade windows
    # overlap (routine here, since a triple-barrier trade can stay open
    # for horizon_bars while a new signal fires every bar). This is the
    # number actually shown to the user, so it must be the honest one.
    valid_idx = np.flatnonzero(label_end_idx >= 0)
    trade_returns = sequential_trade_returns(close, valid_idx, label_end_idx[valid_idx], label_ints[valid_idx], LABEL_TO_INT, cost_bps)
    report = compounded_report(trade_returns)

    label_counts = {name: int((label_ints[valid_idx] == idx).sum()) for name, idx in LABEL_TO_INT.items()}

    return BacktestReport(
        hit_rate=report.win_rate,
        net_pnl=report.total_return,
        sharpe=report.sharpe,
        max_drawdown=report.max_drawdown,
        n_signals=report.n_trades,
        equity_curve=report.equity_curve,
        label_counts=label_counts,
    )

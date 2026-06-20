"""Triple-barrier labeling with volatility-scaled (ATR) barriers.

docs/02 "Target Labeling Strategy": a flat 1% threshold is statistically
wrong across assets of different volatility (mega-cap vs. BIST
small-cap). Each sample instead gets two horizontal barriers scaled by
ATR (`k * ATR`) plus a vertical barrier (the forward horizon, in bars).
If price hits the upper barrier first -> Buy, the lower barrier first
-> Sell, neither within the horizon -> Hold.

This is an **offline, training-time** preprocessing step (not part of
the live inference path in inference/analysis_engine.py), so the
per-sample forward scan here is a plain Python loop rather than a
vectorized op — clarity and correctness matter more than throughput for
a one-time label pass over historical data.

Leakage rule (docs/02, critical): every label's forward window
[i+1, label_end_idx] must be purged from the training features of any
sample whose own window overlaps it, plus an embargo gap after — that
purge/embargo step lives in ml_pipeline/validation.py and consumes the
`label_end_idx` this module returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Default ATR multipliers per horizon bucket. These are *defaults*, not
# hard rules — docs/02 says fixed percentages are retained only as
# sanity bounds; tune k per asset/horizon via the ML pipeline.
HORIZON_DEFAULTS = {
    "short": {"k_upper": 1.0, "k_lower": 1.5},   # 5m-1H
    "medium": {"k_upper": 2.0, "k_lower": 2.0},  # 1D-1W
    "long": {"k_upper": 3.0, "k_lower": 3.0},    # 1M+
}

LABELS = {1: "Buy", 0: "Hold", -1: "Sell"}


def triple_barrier_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    k_upper: float,
    k_lower: float,
    price_col: str = "close",
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """Label each row using volatility-scaled triple-barrier method.

    Returns a DataFrame indexed like `df` with columns:
    - label: "Buy" | "Hold" | "Sell" (NaN where unlabelable)
    - label_end_idx: integer position of the barrier touch (or horizon end)
    - upper_barrier / lower_barrier: the price levels used

    Rows without a full forward window of `horizon_bars` (insufficient
    future data) or without a valid ATR (warm-up) are left unlabeled —
    never guessed as Hold, since that would silently mislabel truncated
    data.
    """
    close = df[price_col].to_numpy()
    atr = df[atr_col].to_numpy()
    n = len(df)

    label = np.full(n, np.nan, dtype=object)
    label_end_idx = np.full(n, -1, dtype=int)
    upper_barrier = np.full(n, np.nan)
    lower_barrier = np.full(n, np.nan)

    last_valid_start = n - 1 - horizon_bars
    for i in range(max(0, last_valid_start) + 1):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry = close[i]
        upper = entry + k_upper * atr[i]
        lower = entry - k_lower * atr[i]
        upper_barrier[i] = upper
        lower_barrier[i] = lower

        end = i + horizon_bars
        outcome = "Hold"
        touch_idx = end
        for j in range(i + 1, end + 1):
            if close[j] >= upper:
                outcome = "Buy"
                touch_idx = j
                break
            if close[j] <= lower:
                outcome = "Sell"
                touch_idx = j
                break

        label[i] = outcome
        label_end_idx[i] = touch_idx

    return pd.DataFrame(
        {
            "label": label,
            "label_end_idx": label_end_idx,
            "upper_barrier": upper_barrier,
            "lower_barrier": lower_barrier,
        },
        index=df.index,
    )


def attach_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    horizon_bucket: str = "medium",
    k_upper: float | None = None,
    k_lower: float | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper: apply HORIZON_DEFAULTS and drop unlabelable rows."""
    defaults = HORIZON_DEFAULTS[horizon_bucket]
    labels_df = triple_barrier_labels(
        df,
        horizon_bars=horizon_bars,
        k_upper=k_upper if k_upper is not None else defaults["k_upper"],
        k_lower=k_lower if k_lower is not None else defaults["k_lower"],
        **kwargs,
    )
    out = df.join(labels_df)
    return out.dropna(subset=["label"])

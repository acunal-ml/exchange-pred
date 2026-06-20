import numpy as np
import pandas as pd

from data_pipeline.feature_engine import atr
from data_pipeline.labeling import attach_labels, label_features, triple_barrier_labels


def _df_with_atr(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")
    close = pd.Series(closes, index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        }
    )
    df["atr_14"] = 1.0  # fixed ATR for deterministic barrier math
    return df


def test_upper_barrier_touch_labels_buy():
    # entry=100, k_upper=1 -> upper barrier=101; price jumps to 105 next bar.
    closes = [100, 105] + [100] * 5
    df = _df_with_atr(closes)
    labels = triple_barrier_labels(df, horizon_bars=3, k_upper=1.0, k_lower=1.0)
    assert labels["label"].iloc[0] == "Buy"
    assert labels["label_end_idx"].iloc[0] == 1


def test_lower_barrier_touch_labels_sell():
    closes = [100, 95] + [100] * 5
    df = _df_with_atr(closes)
    labels = triple_barrier_labels(df, horizon_bars=3, k_upper=1.0, k_lower=1.0)
    assert labels["label"].iloc[0] == "Sell"


def test_no_touch_within_horizon_labels_hold():
    closes = [100, 100.2, 100.3, 100.1] + [100] * 5
    df = _df_with_atr(closes)
    labels = triple_barrier_labels(df, horizon_bars=3, k_upper=1.0, k_lower=1.0)
    assert labels["label"].iloc[0] == "Hold"
    assert labels["label_end_idx"].iloc[0] == 3  # vertical barrier = horizon end


def test_rows_without_full_forward_window_are_unlabeled_not_hold():
    # Critical leakage guard: truncated tail data must NOT be silently
    # labeled "Hold" — that would mislabel samples we simply can't confirm.
    closes = [100] * 10
    df = _df_with_atr(closes)
    labels = triple_barrier_labels(df, horizon_bars=5, k_upper=1.0, k_lower=1.0)
    assert labels["label"].iloc[-1:].isna().all()
    assert labels["label"].iloc[: len(df) - 5].notna().all()


def test_warmup_rows_without_atr_are_unlabeled():
    closes = [100] * 10
    df = _df_with_atr(closes)
    df.loc[df.index[:3], "atr_14"] = np.nan
    labels = triple_barrier_labels(df, horizon_bars=3, k_upper=1.0, k_lower=1.0)
    assert labels["label"].iloc[:3].isna().all()


def test_attach_labels_drops_unlabelable_rows():
    closes = [100 + i * 0.01 for i in range(20)]
    df = _df_with_atr(closes)
    out = attach_labels(df, horizon_bars=5, horizon_bucket="medium")
    assert out["label"].notna().all()
    assert len(out) < len(df)
    assert {"label", "label_end_idx", "upper_barrier", "lower_barrier"}.issubset(out.columns)


def test_label_features_preserves_every_row_and_position():
    # Regression test: label_end_idx is a *positional* index into this
    # same frame. If unlabelable tail rows were dropped (as attach_labels
    # does), an earlier row's label_end_idx pointing at one of those tail
    # positions would go out of bounds — exactly the bug this function
    # exists to avoid for ml_pipeline.validation / train_lightgbm.
    closes = [100 + i * 0.01 for i in range(20)]
    df = _df_with_atr(closes)
    out = label_features(df, horizon_bars=5, horizon_bucket="medium")

    assert len(out) == len(df)
    assert out["label"].isna().any()  # tail rows remain, unlabeled

    valid = out["label_end_idx"] >= 0
    assert (out.loc[valid, "label_end_idx"] < len(out)).all()
    # every referenced exit position must still have real price data
    close_values = out["close"].to_numpy()
    for end_idx in out.loc[valid, "label_end_idx"]:
        assert not np.isnan(close_values[int(end_idx)])

"""Shared utilities between train_lightgbm.py and train_lstm.py.

Factored out so both models' training scripts use the exact same data
build, feature set, seeding, and metric-reporting code — one canonical
path per docs/01's DRY principle, applied here to the training side so
the two models stay comparable (same features, same labels, same CV).
"""
from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from data_pipeline.feature_engine import compute_features, drop_warmup
from data_pipeline.labeling import INT_TO_LABEL, label_features
from data_pipeline.sources.ingest_tvdatafeed import TVDatafeedSource
from data_pipeline.sources.ingest_yfinance import YFinanceSource

FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "bb_percent_b",
    "atr_14",
]

NUM_CLASS = 3  # Sell, Hold, Buy — see data_pipeline.labeling.LABEL_TO_INT


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # LightGBM-only training doesn't need torch installed


def feature_set_hash(feature_columns: list[str]) -> str:
    return hashlib.sha256(",".join(feature_columns).encode()).hexdigest()[:12]


def build_labeled_dataset(
    symbol: str,
    timeframe: str,
    source: str,
    lookback_days: int,
    horizon_bars: int,
    horizon_bucket: str,
) -> pd.DataFrame:
    """Fetch -> feature -> label, reusing the exact same code path as
    inference (docs/01 DRY requirement) so train/serve never diverge.

    Uses `label_features` (not `attach_labels`): every downstream
    consumer indexes `close`/`label_end_idx` by raw position (validation
    folds, financial-metric exit lookups, LSTM sequence windows), so
    unlabelable rows must stay in the frame rather than be dropped — see
    label_features' docstring in data_pipeline.labeling.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    data_source = YFinanceSource() if source == "yfinance" else TVDatafeedSource()
    df = data_source.fetch_ohlcv(symbol, timeframe, start, end)

    feats = drop_warmup(compute_features(df))
    return label_features(feats, horizon_bars=horizon_bars, horizon_bucket=horizon_bucket)


def carve_early_stopping_val(train_idx: np.ndarray, val_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Split a (chronologically sorted) train index into fit/early-stop-val
    using the most recent bars as validation — never a random split."""
    n_val = max(1, int(len(train_idx) * val_frac))
    return train_idx[:-n_val], train_idx[-n_val:]


def log_metrics_safe(metrics: dict[str, float]) -> None:
    """mlflow.log_metrics, dropping NaN/inf values.

    financial_report() legitimately returns NaN sharpe when a holdout has
    zero trades (e.g. a degenerate "always Hold" model) — that's a real,
    informative outcome, not a bug. But MLflow 3.x's NaN metric storage
    collides with mlflow.pytorch.log_model's internal metric snapshotting
    (a UNIQUE-constraint IntegrityError on the metrics table), so non-finite
    values are skipped here rather than passed through to mlflow.log_metrics.
    """
    import mlflow

    finite = {k: v for k, v in metrics.items() if np.isfinite(v)}
    if finite:
        mlflow.log_metrics(finite)


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    report = {
        "f1_macro": f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
    }
    for class_id, name in INT_TO_LABEL.items():
        report[f"precision_{name.lower()}"] = float(precision[class_id])
        report[f"recall_{name.lower()}"] = float(recall[class_id])
        report[f"f1_{name.lower()}"] = float(f1[class_id])
    return report

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from backtest.walk_forward import run_backtest
from ml_pipeline.common import FEATURE_COLUMNS


def _synthetic_ohlcv(n: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1e5, 1e6, n)
    idx = pd.date_range(end=datetime.now(UTC) - timedelta(days=1), periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_run_backtest_indicators_only_returns_report():
    df = _synthetic_ohlcv(n=120)
    report = run_backtest(
        df,
        feature_columns=FEATURE_COLUMNS,
        horizon_bars=10,
        horizon_bucket="medium",
        w_ind=1.0,
        w_lgbm=0.0,
        w_lstm=0.0,
        confidence_threshold=0.4,
    )
    assert isinstance(report.net_pnl, float)
    assert isinstance(report.n_signals, int)
    assert set(report.label_counts.keys()) == {"Sell", "Hold", "Buy"}
    assert report.max_drawdown <= 0.0


def test_run_backtest_high_threshold_yields_mostly_hold():
    df = _synthetic_ohlcv(n=120)
    report = run_backtest(
        df,
        feature_columns=FEATURE_COLUMNS,
        horizon_bars=10,
        horizon_bucket="medium",
        w_ind=1.0,
        w_lgbm=0.0,
        w_lstm=0.0,
        confidence_threshold=0.999,
    )
    assert report.label_counts["Hold"] >= report.label_counts["Buy"]
    assert report.label_counts["Hold"] >= report.label_counts["Sell"]
    assert report.n_signals == 0  # an all-Hold backtest has no trades


def test_run_backtest_with_lightgbm_bundle(tmp_path):
    import joblib
    import lightgbm as lgb

    from inference.model_loader import load_model_bundle
    from ml_pipeline.calibrate import fit_calibrators
    from ml_pipeline.export_onnx import export_lightgbm_to_onnx

    df = _synthetic_ohlcv(n=150)
    rng = np.random.default_rng(0)
    n_features = len(FEATURE_COLUMNS)
    X = rng.normal(0, 1, (200, n_features)).astype(np.float32)
    y = rng.integers(0, 3, 200)
    model = lgb.LGBMClassifier(objective="multiclass", num_class=3, verbosity=-1, n_estimators=10)
    model.fit(X, y)
    export_lightgbm_to_onnx(model, n_features, tmp_path / "model.onnx")
    joblib.dump(fit_calibrators(model.predict_proba(X), y, method="sigmoid"), tmp_path / "calibrators.joblib")

    bundle = load_model_bundle(model_type="lightgbm", local_dir=tmp_path)

    report = run_backtest(
        df,
        feature_columns=FEATURE_COLUMNS,
        horizon_bars=10,
        horizon_bucket="medium",
        w_ind=1.0,
        w_lgbm=1.0,
        w_lstm=0.0,
        confidence_threshold=0.4,
        lgbm_bundle=bundle,
    )
    assert isinstance(report.net_pnl, float)


def test_run_backtest_empty_features_returns_empty_report():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    report = run_backtest(df, feature_columns=FEATURE_COLUMNS, horizon_bars=10)
    assert report.n_signals == 0
    assert report.equity_curve == []

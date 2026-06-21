from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from inference.analysis_engine import analyze, drop_unclosed_candle


def _synthetic_ohlcv(n: int = 100, freq: str = "1D", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1e5, 1e6, n)
    idx = pd.date_range(end=datetime.now(UTC) - timedelta(days=1), periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_drop_unclosed_candle_keeps_closed_bars():
    idx = pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")
    df = pd.DataFrame({"close": range(5)}, index=idx)
    now = idx[-1] + timedelta(days=2)  # last bar long closed
    out = drop_unclosed_candle(df, "1D", now=now)
    assert len(out) == 5


def test_drop_unclosed_candle_drops_still_forming_bar():
    idx = pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")
    df = pd.DataFrame({"close": range(5)}, index=idx)
    now = idx[-1] + timedelta(hours=1)  # last daily bar not closed yet
    out = drop_unclosed_candle(df, "1D", now=now)
    assert len(out) == 4


def test_drop_unclosed_candle_handles_empty_df():
    df = pd.DataFrame(columns=["close"])
    out = drop_unclosed_candle(df, "1D")
    assert out.empty


def test_analyze_with_indicators_only_returns_standardized_result():
    df = _synthetic_ohlcv(n=80)
    result = analyze(
        symbol="SYN",
        market="NASDAQ",
        timeframe="1D",
        confidence_threshold=0.4,
        ohlcv_df=df,
    )
    assert result.label in ("Buy", "Hold", "Sell")
    assert 0.0 <= result.confidence <= 1.0
    assert result.timeframe == "1D"
    assert "final" in result.per_source_probs
    assert result.per_source_probs["lightgbm"] is None
    assert result.per_source_probs["lstm"] is None
    assert "entry" in result.levels


def test_analyze_raises_on_no_closed_candle_data():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    with pytest.raises(ValueError):
        analyze(symbol="SYN", market="NASDAQ", timeframe="1D", ohlcv_df=df)


def test_analyze_with_lightgbm_bundle_included(tmp_path):
    import joblib
    import lightgbm as lgb

    from inference.model_loader import load_model_bundle
    from ml_pipeline.calibrate import fit_calibrators
    from ml_pipeline.common import FEATURE_COLUMNS
    from ml_pipeline.export_onnx import export_lightgbm_to_onnx

    rng = np.random.default_rng(0)
    n_features = len(FEATURE_COLUMNS)
    X = rng.normal(0, 1, (200, n_features)).astype(np.float32)
    y = rng.integers(0, 3, 200)
    model = lgb.LGBMClassifier(objective="multiclass", num_class=3, verbosity=-1, n_estimators=10)
    model.fit(X, y)
    export_lightgbm_to_onnx(model, n_features, tmp_path / "model.onnx")
    joblib.dump(fit_calibrators(model.predict_proba(X), y, method="sigmoid"), tmp_path / "calibrators.joblib")

    bundle = load_model_bundle(model_type="lightgbm", local_dir=tmp_path)

    df = _synthetic_ohlcv(n=80)
    result = analyze(
        symbol="SYN",
        market="NASDAQ",
        timeframe="1D",
        ohlcv_df=df,
        lgbm_bundle=bundle,
        w_lgbm=1.0,
        w_ind=1.0,
    )
    assert result.per_source_probs["lightgbm"] is not None
    assert len(result.per_source_probs["lightgbm"]) == 3

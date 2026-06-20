import numpy as np
import pandas as pd
import pytest

from ml_pipeline import train_lightgbm as tl


def _synthetic_labeled_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")

    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(index=idx)
    df["close"] = close
    for col in tl.FEATURE_COLUMNS:
        df[col] = rng.normal(0, 1, n)

    labels = rng.choice(["Buy", "Hold", "Sell"], size=n, p=[0.3, 0.4, 0.3])
    df["label"] = labels
    df["label_end_idx"] = np.minimum(np.arange(n) + 5, n - 1)
    df["upper_barrier"] = close + 1
    df["lower_barrier"] = close - 1
    return df


@pytest.fixture
def mlflow_tmp_tracking(tmp_path, monkeypatch):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    yield


def test_classification_report_dict_has_macro_f1_and_per_class():
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    report = tl.classification_report_dict(y_true, y_pred)
    assert "f1_macro" in report
    assert "precision_buy" in report and "recall_sell" in report and "f1_hold" in report


def test_feature_set_hash_is_stable_and_order_sensitive():
    h1 = tl.feature_set_hash(["a", "b", "c"])
    h2 = tl.feature_set_hash(["a", "b", "c"])
    h3 = tl.feature_set_hash(["c", "b", "a"])
    assert h1 == h2
    assert h1 != h3


def test_carve_early_stopping_val_uses_chronological_tail():
    train_idx = np.arange(20)
    fit_idx, es_idx = tl.carve_early_stopping_val(train_idx, val_frac=0.2)
    assert fit_idx.max() < es_idx.min()
    assert len(es_idx) == 4


def test_run_training_end_to_end_on_synthetic_data(mlflow_tmp_tracking):
    df = _synthetic_labeled_df(n=300)
    run_id = tl.run_training(
        symbol="SYN",
        timeframe="1D",
        n_splits=3,
        n_trials=2,
        embargo_bars=5,
        holdout_frac=0.2,
        seed=7,
        experiment_name="test_synthetic",
        labeled_df=df,
    )
    assert run_id is not None

    import mlflow

    run = mlflow.get_run(run_id)
    assert "f1_macro" in run.data.metrics
    assert "holdout_net_pnl" in run.data.metrics
    assert "baseline_buy_and_hold_net_pnl" in run.data.metrics

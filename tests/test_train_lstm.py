import numpy as np
import pandas as pd
import pytest

from ml_pipeline import train_lstm as tl


def _synthetic_labeled_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
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
def mlflow_tmp_tracking(tmp_path):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    yield


def test_build_splits_excludes_positions_without_full_sequence_history():
    n = 200
    label_end_idx = np.minimum(np.arange(n) + 5, n - 1)
    holdout_fold, folds = tl._build_splits(n, label_end_idx, seq_len=20, n_splits=3, embargo_bars=5, holdout_frac=0.2)
    all_idx = np.concatenate([holdout_fold.train_idx, holdout_fold.test_idx] + [f.train_idx for f in folds] + [f.test_idx for f in folds])
    assert (all_idx >= 19).all()  # seq_len - 1


def test_train_one_fold_runs_on_cpu_and_returns_predictions():
    n, n_features, seq_len = 150, len(tl.FEATURE_COLUMNS), 10
    rng = np.random.default_rng(1)
    X_full = rng.normal(0, 1, (n, n_features)).astype(np.float32)
    y_full = rng.integers(0, 3, n)
    label_end_idx = np.minimum(np.arange(n) + 5, n - 1)

    from ml_pipeline.validation import purged_walk_forward_splits

    eligible = tl._lstm_eligible_label_end_idx(label_end_idx, seq_len)
    folds = list(purged_walk_forward_splits(n, eligible, n_splits=3, embargo_bars=2))
    assert folds

    params = {
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "grad_accum_steps": 1,
        "grad_clip_norm": 1.0,
    }
    model, scaler, y_pred, y_proba = tl.train_one_fold(
        params, X_full, y_full, folds[-1], seq_len, seed=0, device="cpu", max_epochs=2, patience=2
    )
    assert y_pred.shape[0] == len(folds[-1].test_idx)
    assert y_proba.shape == (len(folds[-1].test_idx), 3)


def test_run_training_end_to_end_on_synthetic_data(mlflow_tmp_tracking):
    df = _synthetic_labeled_df(n=400)
    run_id = tl.run_training(
        symbol="SYN",
        timeframe="1D",
        n_splits=2,
        n_trials=2,
        embargo_bars=5,
        holdout_frac=0.2,
        seed=7,
        experiment_name="test_synthetic_lstm",
        tuning_max_epochs=2,
        tuning_patience=1,
        final_max_epochs=2,
        final_patience=1,
        labeled_df=df,
    )
    assert run_id is not None

    import mlflow

    run = mlflow.get_run(run_id)
    assert "f1_macro" in run.data.metrics
    assert "holdout_net_pnl" in run.data.metrics
    assert "baseline_buy_and_hold_net_pnl" in run.data.metrics

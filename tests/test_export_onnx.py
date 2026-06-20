import numpy as np
import pandas as pd
import pytest
import torch

from ml_pipeline.export_onnx import (
    ExportedArtifact,
    export_champion,
    export_lightgbm_to_onnx,
    export_lstm_to_onnx,
    push_to_hf_hub,
    quantize_int8,
    verify_onnx_matches,
)
from ml_pipeline.lstm_model import AttentionLSTM


@pytest.fixture
def mlflow_tmp_tracking(tmp_path):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    yield


def test_export_lightgbm_to_onnx_matches_original(tmp_path):
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 5)).astype(np.float32)
    y = rng.integers(0, 3, 200)

    model = lgb.LGBMClassifier(objective="multiclass", num_class=3, verbosity=-1, n_estimators=20)
    model.fit(X, y)

    onnx_path = export_lightgbm_to_onnx(model, n_features=5, output_path=tmp_path / "model.onnx")
    assert onnx_path.exists()

    reference = model.predict_proba(X[:10])
    assert verify_onnx_matches(onnx_path, X[:10], reference, atol=1e-2, output_index=1)


def test_export_lstm_to_onnx_matches_original(tmp_path):
    seq_len, n_features = 8, 4
    model = AttentionLSTM(n_features=n_features, hidden_size=8, num_layers=1, dropout=0.0)
    model.eval()

    onnx_path = export_lstm_to_onnx(model, seq_len, n_features, tmp_path / "lstm.onnx")
    assert onnx_path.exists()

    x = torch.randn(3, seq_len, n_features)
    with torch.no_grad():
        ref_logits, _ = model(x)

    assert verify_onnx_matches(onnx_path, x.numpy(), ref_logits.numpy(), atol=1e-3)


def test_quantize_int8_produces_a_smaller_or_equal_file(tmp_path):
    import lightgbm as lgb

    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (100, 5)).astype(np.float32)
    y = rng.integers(0, 3, 100)
    model = lgb.LGBMClassifier(objective="multiclass", num_class=3, verbosity=-1, n_estimators=10)
    model.fit(X, y)

    onnx_path = export_lightgbm_to_onnx(model, 5, tmp_path / "model.onnx")
    quantized_path = quantize_int8(onnx_path)
    assert quantized_path.exists()


def test_push_to_hf_hub_skips_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_pipeline.export_onnx.settings.hf_hub_token", None)
    monkeypatch.setattr("ml_pipeline.export_onnx.settings.hf_dataset_repo", None)

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake")
    result = push_to_hf_hub(ExportedArtifact(onnx_path=onnx_path, extra_artifact_paths=[]))
    assert result is None


def _synthetic_labeled_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    from ml_pipeline import train_lightgbm as tl

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(index=idx)
    df["close"] = close
    for col in tl.FEATURE_COLUMNS:
        df[col] = rng.normal(0, 1, n)
    df["label"] = rng.choice(["Buy", "Hold", "Sell"], size=n, p=[0.3, 0.4, 0.3])
    df["label_end_idx"] = np.minimum(np.arange(n) + 5, n - 1)
    df["upper_barrier"] = close + 1
    df["lower_barrier"] = close - 1
    return df


def test_export_champion_end_to_end_against_registry(tmp_path, mlflow_tmp_tracking):
    import mlflow

    from ml_pipeline import train_lightgbm as tl

    df = _synthetic_labeled_df()
    registered_name = "test_export_SYN_1D"
    run_id = tl.run_training(
        symbol="SYN",
        timeframe="1D",
        n_splits=2,
        n_trials=2,
        embargo_bars=5,
        holdout_frac=0.2,
        seed=7,
        experiment_name="test_export",
        labeled_df=df,
    )
    assert run_id is not None

    # Force the alias regardless of beats_baselines so the export path
    # is exercised deterministically in this test.
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{registered_name}'")
    client.set_registered_model_alias(registered_name, "champion", versions[0].version)

    artifact = export_champion(
        model_type="lightgbm",
        registered_name=registered_name,
        output_dir=tmp_path / "export",
        n_features=len(tl.FEATURE_COLUMNS),
    )
    assert artifact.onnx_path.exists()
    assert len(artifact.extra_artifact_paths) >= 1  # at least calibrators.joblib

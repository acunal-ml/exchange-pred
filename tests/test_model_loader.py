import numpy as np
import pytest

from inference.model_loader import ensure_local_artifacts, load_model_bundle, predict_proba
from ml_pipeline.calibrate import fit_calibrators
from ml_pipeline.export_onnx import export_lightgbm_to_onnx, export_lstm_to_onnx
from ml_pipeline.lstm_model import AttentionLSTM


def test_load_lightgbm_bundle_and_predict(tmp_path):
    import joblib
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 5)).astype(np.float32)
    y = rng.integers(0, 3, 200)

    model = lgb.LGBMClassifier(objective="multiclass", num_class=3, verbosity=-1, n_estimators=15)
    model.fit(X, y)
    export_lightgbm_to_onnx(model, n_features=5, output_path=tmp_path / "model.onnx")

    calib = fit_calibrators(model.predict_proba(X), y, method="sigmoid")
    joblib.dump(calib, tmp_path / "calibrators.joblib")

    bundle = load_model_bundle(model_type="lightgbm", local_dir=tmp_path)
    assert bundle.scaler is None
    assert bundle.calibrators is not None

    proba = predict_proba(bundle, X[:5])
    assert proba.shape == (5, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_load_lstm_bundle_and_predict(tmp_path):
    import joblib
    from sklearn.preprocessing import StandardScaler

    seq_len, n_features = 6, 4
    model = AttentionLSTM(n_features=n_features, hidden_size=8, num_layers=1, dropout=0.0)
    model.eval()
    export_lstm_to_onnx(model, seq_len, n_features, tmp_path / "model.onnx")

    scaler = StandardScaler().fit(np.random.default_rng(0).normal(0, 1, (50, n_features)))
    joblib.dump(scaler, tmp_path / "scaler.joblib")

    bundle = load_model_bundle(model_type="lstm", local_dir=tmp_path, seq_len=seq_len)
    assert bundle.scaler is not None
    assert bundle.calibrators is None

    X = np.random.default_rng(1).normal(0, 1, (3, seq_len, n_features)).astype(np.float32)
    proba = predict_proba(bundle, X)
    assert proba.shape == (3, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_load_lstm_bundle_without_seq_len_raises(tmp_path):
    model = AttentionLSTM(n_features=3, hidden_size=4, num_layers=1, dropout=0.0)
    model.eval()
    export_lstm_to_onnx(model, seq_len=5, n_features=3, output_path=tmp_path / "model.onnx")

    with pytest.raises(ValueError):
        load_model_bundle(model_type="lstm", local_dir=tmp_path, seq_len=None)


def test_ensure_local_artifacts_raises_without_model_or_hf_repo(tmp_path, monkeypatch):
    # Isolate from the real local .env, which has a genuine HF_DATASET_REPO
    # configured for this project's actual deployment.
    monkeypatch.setattr("inference.model_loader.settings.hf_dataset_repo", None)
    with pytest.raises(FileNotFoundError):
        ensure_local_artifacts(tmp_path, hf_dataset_repo=None)

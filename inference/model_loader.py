"""Loads an exported ONNX champion (+ scaler/calibrators) and exposes a
single `predict_proba` regardless of whether it's the LightGBM or the
LSTM model — analysis_engine.py never needs to know which.

Artifacts come from ml_pipeline/export_onnx.py's output directory
(model.onnx, optionally scaler.joblib, optionally calibrators.joblib).
If that directory is empty/missing and an HF dataset repo is configured,
pulls it down once via huggingface_hub and caches it locally — this is
the "Space only loads and runs this artifact, it never trains" boundary
from docs/03.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

from config.settings import settings
from ml_pipeline.calibrate import CalibrationResult, apply_calibration
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelBundle:
    model_type: str  # "lightgbm" | "lstm"
    session: ort.InferenceSession
    scaler: object | None = None  # StandardScaler, LSTM only
    calibrators: CalibrationResult | None = None
    seq_len: int | None = None  # LSTM only


def ensure_local_artifacts(local_dir: Path, hf_dataset_repo: str | None = None, path_in_repo: str = "") -> Path:
    """Make sure model.onnx exists under local_dir, downloading from the
    configured HF dataset repo on first use if it's missing."""
    local_dir = Path(local_dir)
    if (local_dir / "model.onnx").exists():
        return local_dir

    repo = hf_dataset_repo or settings.hf_dataset_repo
    if not repo:
        raise FileNotFoundError(f"No model.onnx in {local_dir} and no HF_DATASET_REPO configured to fetch it from.")

    from huggingface_hub import snapshot_download

    logger.info("Downloading model artifacts from HF dataset repo %s", repo)
    downloaded = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=[f"{path_in_repo}*"] if path_in_repo else None,
        token=settings.hf_hub_token or None,
    )
    return Path(downloaded) / path_in_repo if path_in_repo else Path(downloaded)


def load_model_bundle(
    model_type: str | None = None,
    local_dir: Path = None,
    seq_len: int | None = None,
    hf_dataset_repo: str | None = None,
    path_in_repo: str = "",
) -> ModelBundle:
    resolved_dir = ensure_local_artifacts(local_dir, hf_dataset_repo, path_in_repo)

    meta_path = resolved_dir / "meta.json"
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text())
        model_type = model_type or meta.get("model_type")
        seq_len = seq_len or meta.get("seq_len")

    if model_type is None:
        raise ValueError("model_type is required (no meta.json found to infer it from)")

    session = ort.InferenceSession(str(resolved_dir / "model.onnx"), providers=["CPUExecutionProvider"])

    scaler_path = resolved_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    calibrators_path = resolved_dir / "calibrators.joblib"
    calibrators = joblib.load(calibrators_path) if calibrators_path.exists() else None

    if model_type == "lstm" and seq_len is None:
        raise ValueError("seq_len is required to load an LSTM model bundle")

    return ModelBundle(model_type=model_type, session=session, scaler=scaler, calibrators=calibrators, seq_len=seq_len)


def try_load_model_bundle(model_type: str, local_dir: Path, **kwargs) -> ModelBundle | None:
    """Like load_model_bundle, but returns None instead of raising when
    no artifact is available locally or on the HF Hub — callers (the UI)
    must degrade to indicator-only fusion rather than crash when a given
    (symbol, timeframe) has no trained champion yet."""
    try:
        return load_model_bundle(model_type, local_dir, **kwargs)
    except (FileNotFoundError, ValueError) as exc:
        logger.info("No %s model bundle available at %s (%s)", model_type, local_dir, exc)
        return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_proba(bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    """Calibrated class probabilities for `X`.

    LightGBM: X is [n_samples, n_features] -> returns [n_samples, 3].
    LSTM: X is [n_samples, seq_len, n_features] (already-scaled windows,
    same scaler this bundle was trained with — callers must scale via
    `bundle.scaler` before calling) -> returns [n_samples, 3].
    """
    input_name = bundle.session.get_inputs()[0].name

    if bundle.model_type == "lightgbm":
        outputs = bundle.session.run(None, {input_name: X.astype(np.float32)})
        raw_proba = outputs[1]  # see export_onnx.py: zipmap=False -> [label, probabilities]
    elif bundle.model_type == "lstm":
        logits = bundle.session.run(None, {input_name: X.astype(np.float32)})[0]
        raw_proba = _softmax(logits)
    else:
        raise ValueError(f"Unknown model_type: {bundle.model_type}")

    if bundle.calibrators is not None:
        return apply_calibration(bundle.calibrators, raw_proba)
    return raw_proba

"""Train -> serve handoff: export the registered "champion" to ONNX.

docs/03: "The champion is exported to ONNX (and optionally
int8-quantized) and pushed to an HF Hub / Dataset repo. The HF Space
only loads and runs this artifact — it never trains." This is the only
module that crosses that boundary: everything upstream (train_lightgbm,
train_lstm, calibrate) is local-only; everything downstream
(inference/analysis_engine.py, to be built) only ever reads what this
module produces.

Quantization caveat: dynamic int8 quantization (`onnxruntime.quantization
.quantize_dynamic`) targets MatMul/LSTM/Gemm-style ops. It meaningfully
shrinks and speeds up the LSTM graph, but a converted LightGBM graph is
almost entirely a single TreeEnsembleClassifier op with no quantizable
weights — running it through quantize_dynamic is harmless but won't do
much. Both paths still go through the same function for one code path,
the LightGBM case just won't see a size/latency win.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
from onnxmltools.convert.common.data_types import FloatTensorType

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExportedArtifact:
    onnx_path: Path
    extra_artifact_paths: list[Path]  # scaler.joblib, calibrators.joblib, etc.


def export_lightgbm_to_onnx(model, n_features: int, output_path: Path) -> Path:
    """Convert a fitted LGBMClassifier to ONNX via onnxmltools."""
    from onnxmltools import convert_lightgbm

    # zipmap=False: without it, the "probabilities" output is a
    # sequence-of-maps (one dict per row), not a dense float tensor —
    # onnxruntime's InferenceSession output is awkward to consume that
    # way, and the inference engine wants a plain [batch, n_classes] array.
    onnx_model = convert_lightgbm(model, initial_types=[("input", FloatTensorType([None, n_features]))], zipmap=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, str(output_path))
    return output_path


def export_lstm_to_onnx(model, seq_len: int, n_features: int, output_path: Path) -> Path:
    """Convert a trained AttentionLSTM to ONNX via torch.onnx.export.

    Exports with a dynamic batch axis so the inference engine can run
    arbitrary batch sizes; seq_len and n_features stay fixed (the model
    architecture itself is sized to them).
    """
    import torch

    model = model.to("cpu").eval()
    dummy_input = torch.zeros(1, seq_len, n_features, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["input"],
        output_names=["logits", "attn_weights"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "attn_weights": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # the default dynamo-based exporter needs onnxscript; not a project dependency
    )
    return output_path


def quantize_int8(onnx_path: Path, output_path: Path | None = None) -> Path:
    """Dynamic int8 quantization — see module docstring for the
    LightGBM-vs-LSTM caveat."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_path = output_path or onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    quantize_dynamic(model_input=str(onnx_path), model_output=str(output_path), weight_type=QuantType.QInt8)
    return output_path


def verify_onnx_matches(
    onnx_path: Path, sample_input: np.ndarray, reference_output: np.ndarray, atol: float = 1e-3, output_index: int = 0
) -> bool:
    """Sanity check: ONNX Runtime's output on a sample must match the
    original model's output within tolerance, before anything ships.

    `output_index`: LightGBM's converted graph exposes `[label, probabilities]`
    — pass 1 to compare against predict_proba. The LSTM graph's first
    output is `logits`, so the default 0 is correct there.
    """
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: sample_input.astype(np.float32)})[output_index]
    return np.allclose(onnx_output, reference_output, atol=atol)


def _resolve_champion_run(registered_name: str) -> tuple[str, str]:
    """Return (run_id, model_uri) for the version aliased 'champion'."""
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(registered_name, "champion")
    return mv.run_id, f"models:/{registered_name}@champion"


def _download_run_artifact(run_id: str, artifact_path: str, dest_dir: Path) -> Path | None:
    try:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path, dst_path=str(dest_dir))
        return Path(local_path)
    except Exception as exc:
        logger.warning("No %s artifact on run %s (%s) — skipping", artifact_path, run_id, exc)
        return None


def export_champion(
    model_type: str,
    registered_name: str,
    output_dir: Path,
    n_features: int,
    seq_len: int | None = None,
    quantize: bool = False,
) -> ExportedArtifact:
    """Load whichever model version holds the 'champion' alias and
    export it + its scaler/calibrators to `output_dir`.

    `model_type` is "lightgbm" or "lstm" — the caller already knows
    which pipeline produced the registered model; this module doesn't
    try to detect it from MLflow's flavor metadata.
    """
    run_id, model_uri = _resolve_champion_run(registered_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "model.onnx"
    if model_type == "lightgbm":
        model = mlflow.lightgbm.load_model(model_uri)
        export_lightgbm_to_onnx(model, n_features, onnx_path)
    elif model_type == "lstm":
        if seq_len is None:
            raise ValueError("seq_len is required to export an LSTM model")
        model = mlflow.pytorch.load_model(model_uri)
        export_lstm_to_onnx(model, seq_len, n_features, onnx_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if quantize:
        onnx_path = quantize_int8(onnx_path)

    extra_paths = []
    for artifact_name in ("scaler.joblib", "calibrators.joblib"):
        downloaded = _download_run_artifact(run_id, artifact_name, output_dir)
        if downloaded is not None:
            extra_paths.append(downloaded)

    # meta.json lets model_loader.load_model_bundle() resolve model_type/
    # seq_len/n_features without the caller having to hardcode them again
    # at serve time — one source of truth for "what shape does this
    # artifact expect", written once here at export time.
    import json

    meta_path = output_dir / "meta.json"
    meta_path.write_text(
        json.dumps({"model_type": model_type, "registered_name": registered_name, "n_features": n_features, "seq_len": seq_len})
    )
    extra_paths.append(meta_path)

    logger.info("Exported champion %s -> %s (+%d extra artifacts)", registered_name, onnx_path, len(extra_paths))
    return ExportedArtifact(onnx_path=onnx_path, extra_artifact_paths=extra_paths)


def push_to_hf_hub(artifact: ExportedArtifact, path_in_repo: str = "") -> str | None:
    """Upload the exported artifact directory to the configured HF
    Dataset repo. Returns the repo URL, or None if no repo is configured
    (this is expected in local dev — the export still succeeds locally,
    it just isn't published) — see docs/04: artifacts are pulled from an
    HF Hub/Dataset repo at Space startup, never committed into the Space
    repo itself.

    `HF_HUB_TOKEN` is optional here: if unset, huggingface_hub's HfApi
    falls back to the credentials cached by `hf auth login` (or the
    `HF_TOKEN` env var) — only the target repo actually needs to be
    configured.
    """
    if not settings.hf_dataset_repo:
        logger.warning(
            "HF_DATASET_REPO not configured — skipping upload, artifact stays local at %s",
            artifact.onnx_path.parent,
        )
        return None

    from huggingface_hub import HfApi

    api = HfApi(token=settings.hf_hub_token or None)
    api.upload_folder(
        folder_path=str(artifact.onnx_path.parent),
        repo_id=settings.hf_dataset_repo,
        repo_type="dataset",
        path_in_repo=path_in_repo,
    )
    repo_url = f"https://huggingface.co/datasets/{settings.hf_dataset_repo}"
    logger.info("Pushed %s to %s", artifact.onnx_path.parent, repo_url)
    return repo_url


def clean_output_dir(output_dir: Path) -> None:
    """Remove a previous export before writing a new one — avoids
    silently mixing artifacts from two different champion versions."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

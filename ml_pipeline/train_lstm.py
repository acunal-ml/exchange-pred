"""Attention-LSTM training: Optuna hyperparameter search + MLflow tracking.

Implements docs/03's LSTM section end to end, mirroring train_lightgbm.py's
overall shape (Optuna under purged/embargoed walk-forward CV, MLflow
nested-run-per-trial, never-touched holdout, baseline-gated "champion"
alias) so the two models are directly comparable. See train_lightgbm.py's
module docstring for the shared MLOps decisions (mlflow stage->alias,
sqlite tracking store requirement, ONNX export deferred to
export_onnx.py) — not repeated here.

What's specific to the LSTM (docs/03 "Resource management (must)" for
the 4GB-VRAM dev box):
- Mixed precision (AMP) via torch.amp.autocast/GradScaler — CUDA only;
  silently disabled on CPU (e.g. CI/tests without a GPU).
- Gradient accumulation: backward() every step, optimizer.step() every
  `grad_accum_steps` steps, so a larger *effective* batch fits in 4GB
  without materializing it at once.
- Gradient clipping (LSTMs are prone to exploding gradients).
- seq_len is itself a tuned hyperparameter — varies the valid sample
  positions per trial, so the walk-forward/holdout split is rebuilt per
  trial (see `_build_splits`) rather than computed once up front like
  train_lightgbm.py does.
- The feature scaler is fit only on each fold's training rows (never on
  test/holdout) and saved as an MLflow artifact alongside the champion
  model — the same scaler must be applied at inference (train/serve
  skew here is a classic silent bug, per docs/03).
"""

from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.labeling import LABEL_TO_INT
from ml_pipeline.calibrate import apply_calibration, expected_calibration_error, fit_calibrators
from ml_pipeline.common import (
    FEATURE_COLUMNS,
    NUM_CLASS,
    build_labeled_dataset,
    carve_early_stopping_val,
    classification_report_dict,
    feature_set_hash,
    log_metrics_safe,
    set_global_seed,
)
from ml_pipeline.eval_plots import plot_calibration_curve, plot_confusion_matrix, plot_equity_curve
from ml_pipeline.financial_metrics import (
    DEFAULT_COST_BPS,
    buy_and_hold_baseline,
    compounded_equity_curve,
    compounded_report,
    financial_report,
    majority_class_baseline,
    sequential_trade_returns,
    strategy_returns,
)
from ml_pipeline.lstm_model import AttentionLSTM
from ml_pipeline.sequence_dataset import SequenceWindowDataset
from ml_pipeline.validation import Fold, out_of_time_holdout_split, purged_walk_forward_splits
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _lstm_eligible_label_end_idx(label_end_idx: np.ndarray, seq_len: int) -> np.ndarray:
    """Mark positions without `seq_len` bars of history as invalid (-1)
    too, on top of the existing label validity — reuses validation.py's
    valid_mask (label_end_idx >= 0) unmodified for the seq_len constraint."""
    adjusted = label_end_idx.copy()
    if seq_len > 1:
        adjusted[: seq_len - 1] = -1
    return adjusted


def _build_splits(
    n_total: int,
    label_end_idx: np.ndarray,
    seq_len: int,
    n_splits: int,
    embargo_bars: int,
    holdout_frac: float,
) -> tuple[Fold, list[Fold]]:
    eligible = _lstm_eligible_label_end_idx(label_end_idx, seq_len)
    holdout_fold = out_of_time_holdout_split(n_total, eligible, holdout_frac, embargo_bars)
    n_tuning = int(holdout_fold.test_idx.min())
    folds = list(purged_walk_forward_splits(n_tuning, eligible[:n_tuning], n_splits=n_splits, embargo_bars=embargo_bars))
    return holdout_fold, folds


def _make_loader(dataset: SequenceWindowDataset, batch_size: int, shuffle: bool, device: str, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )


def train_one_fold(
    params: dict,
    X_full: np.ndarray,
    y_full: np.ndarray,
    fold: Fold,
    seq_len: int,
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, StandardScaler, np.ndarray, np.ndarray]:
    """Train on fold.train_idx (chronological-tail early stopping),
    return the early-stopped model + fitted scaler + predictions/probas
    on fold.test_idx."""
    fit_idx, es_idx = carve_early_stopping_val(fold.train_idx)

    scaler = StandardScaler().fit(X_full[fit_idx])
    scaled = scaler.transform(X_full).astype(np.float32)

    train_ds = SequenceWindowDataset(scaled, y_full, fit_idx, seq_len)
    es_ds = SequenceWindowDataset(scaled, y_full, es_idx, seq_len)
    test_ds = SequenceWindowDataset(scaled, y_full, fold.test_idx, seq_len)

    train_loader = _make_loader(train_ds, params["batch_size"], shuffle=True, device=device)
    es_loader = _make_loader(es_ds, params["batch_size"], shuffle=False, device=device)
    test_loader = _make_loader(test_ds, params["batch_size"], shuffle=False, device=device)

    class_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASS), y=y_full[fit_idx])
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    torch.manual_seed(seed)
    model = AttentionLSTM(
        n_features=X_full.shape[1],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        num_classes=NUM_CLASS,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    use_amp = device == "cuda"
    amp_scaler = torch.amp.GradScaler(device, enabled=use_amp)
    grad_accum_steps = params.get("grad_accum_steps", 1)
    grad_clip_norm = params.get("grad_clip_norm", 1.0)

    best_es_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for _epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast(device, enabled=use_amp):
                logits, _ = model(xb)
                loss = criterion(logits, yb) / grad_accum_steps
            amp_scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0:
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad()

        model.eval()
        es_loss_total, n_batches = 0.0, 0
        with torch.no_grad():
            for xb, yb in es_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb)
                es_loss_total += criterion(logits, yb).item()
                n_batches += 1
        es_loss = es_loss_total / max(n_batches, 1)

        if es_loss < best_es_loss - 1e-4:
            best_es_loss = es_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_pred, y_proba = _predict_loader(model, test_loader, device)
    return model, scaler, y_pred, y_proba


def _predict_loader(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_proba = [], []
    with torch.no_grad():
        for xb, _yb in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(proba.argmax(axis=1))
            all_proba.append(proba)

    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    y_proba = np.concatenate(all_proba) if all_proba else np.zeros((0, NUM_CLASS))
    return y_pred, y_proba


def predict_positions(
    model: nn.Module,
    scaler: StandardScaler,
    X_full: np.ndarray,
    positions: np.ndarray,
    seq_len: int,
    y_full: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the trained model + its fold-fit scaler over an arbitrary
    position set (e.g. a calibration slice) — used outside train_one_fold
    so calibration doesn't require retraining."""
    scaled = scaler.transform(X_full).astype(np.float32)
    ds = SequenceWindowDataset(scaled, y_full, positions, seq_len)
    loader = _make_loader(ds, batch_size, shuffle=False, device=device)
    return _predict_loader(model, loader, device)


def _collect_mean_attention(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray | None:
    model.eval()
    weights = []
    with torch.no_grad():
        for xb, _yb in loader:
            xb = xb.to(device)
            _, attn = model(xb)
            weights.append(attn.cpu().numpy())
    if not weights:
        return None
    return np.concatenate(weights, axis=0).mean(axis=0)


def _plot_attention_weights(mean_attn: np.ndarray, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(mean_attn)), mean_attn)
    ax.set_xlabel("Time step (0 = oldest in window)")
    ax.set_ylabel("Mean attention weight")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_training(
    symbol: str,
    timeframe: str,
    source: str = "yfinance",
    lookback_days: int = 900,
    horizon_bars: int = 10,
    horizon_bucket: str = "medium",
    n_splits: int = 5,
    n_trials: int = 20,
    embargo_bars: int = 10,
    holdout_frac: float = 0.15,
    cost_bps: float = DEFAULT_COST_BPS,
    objective_metric: str = "f1_macro",
    seed: int = 42,
    experiment_name: str = "lstm_buy_hold_sell",
    tuning_max_epochs: int = 15,
    tuning_patience: int = 4,
    final_max_epochs: int = 40,
    final_patience: int = 6,
    calibration_method: str = "isotonic",
    calibration_frac: float = 0.15,
    labeled_df: pd.DataFrame | None = None,
) -> str:
    """Full train+tune+evaluate+register pipeline for one (symbol, timeframe).

    `labeled_df` is a testing hook (see train_lightgbm.run_training).

    A calibration slice is carved from the champion's training data,
    chronologically between the early-stopping slice and the final
    holdout (fit < early-stop-val < calibration < holdout) — see
    train_lightgbm.run_training's docstring for why.

    Returns the MLflow run ID of the parent (champion) run.
    """
    set_global_seed(seed)
    mlflow.set_experiment(experiment_name)
    device = _device()
    logger.info("Training on device=%s", device)

    labeled = (
        labeled_df
        if labeled_df is not None
        else build_labeled_dataset(symbol, timeframe, source, lookback_days, horizon_bars, horizon_bucket)
    )
    if len(labeled) < 100:
        raise ValueError(f"Not enough labeled samples ({len(labeled)}) to train an LSTM on")

    X_full = labeled[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_full = labeled["label"].map(LABEL_TO_INT).fillna(-1).astype(int).to_numpy()
    close = labeled["close"].to_numpy()
    label_end_idx = labeled["label_end_idx"].to_numpy()
    n_total = len(labeled)

    with mlflow.start_run(run_name=f"{symbol}_{timeframe}_lstm") as parent_run:
        mlflow.log_params(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "horizon_bars": horizon_bars,
                "horizon_bucket": horizon_bucket,
                "n_splits": n_splits,
                "embargo_bars": embargo_bars,
                "holdout_frac": holdout_frac,
                "cost_bps": cost_bps,
                "objective_metric": objective_metric,
                "seed": seed,
                "device": device,
                "feature_set_hash": feature_set_hash(FEATURE_COLUMNS),
                "data_start": str(labeled.index.min()),
                "data_end": str(labeled.index.max()),
                "n_samples": n_total,
            }
        )

        def objective(trial: optuna.Trial) -> float:
            seq_len = trial.suggest_int("seq_len", 10, 40)
            params = {
                "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                "grad_accum_steps": trial.suggest_categorical("grad_accum_steps", [1, 2, 4]),
                "grad_clip_norm": 1.0,
            }

            _holdout_fold, folds = _build_splits(n_total, label_end_idx, seq_len, n_splits, embargo_bars, holdout_frac)
            if not folds:
                return float("-inf")

            fold_scores = []
            with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
                mlflow.log_params({**params, "seq_len": seq_len, "seed": seed})

                for fold_i, fold in enumerate(folds):
                    if len(fold.test_idx) == 0:
                        continue
                    _model, _scaler, preds, _proba = train_one_fold(
                        params, X_full, y_full, fold, seq_len, seed, device, tuning_max_epochs, tuning_patience
                    )
                    if len(preds) == 0:
                        score = float("-inf")
                    elif objective_metric == "financial":
                        returns = strategy_returns(close, fold.test_idx, label_end_idx[fold.test_idx], preds, LABEL_TO_INT, cost_bps)
                        score = financial_report(returns).net_pnl
                    else:
                        score = classification_report_dict(y_full[fold.test_idx], preds)["f1_macro"]

                    mlflow.log_metric(f"fold{fold_i}_{objective_metric}", score)
                    fold_scores.append(score)

                mean_score = float(np.nanmean(fold_scores)) if fold_scores else float("-inf")
                mlflow.log_metric(f"mean_{objective_metric}", mean_score)

            return mean_score

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)

        best = dict(study.best_params)
        seq_len = best.pop("seq_len")
        best["grad_clip_norm"] = 1.0
        mlflow.log_params({f"best_{k}": v for k, v in {**best, "seq_len": seq_len}.items()})
        mlflow.log_metric(f"best_{objective_metric}", study.best_value)

        holdout_fold, _tuning_folds = _build_splits(n_total, label_end_idx, seq_len, n_splits, embargo_bars, holdout_frac)

        # Chronological order: fit < early-stop-val (inside train_one_fold)
        # < calibration < holdout. The calibration slice is held out of
        # champion training entirely, then used once, post-hoc, to fit
        # the probability calibrators.
        champion_train_idx, calibration_idx = carve_early_stopping_val(holdout_fold.train_idx, val_frac=calibration_frac)
        calibration_fold = Fold(train_idx=champion_train_idx, test_idx=holdout_fold.test_idx)

        champion, scaler, y_pred, y_proba_raw = train_one_fold(
            best, X_full, y_full, calibration_fold, seq_len, seed, device, final_max_epochs, final_patience
        )

        _cal_pred, cal_proba_raw = predict_positions(champion, scaler, X_full, calibration_idx, seq_len, y_full, best["batch_size"], device)
        calibration_result = fit_calibrators(cal_proba_raw, y_full[calibration_idx], method=calibration_method)
        y_proba = apply_calibration(calibration_result, y_proba_raw)

        holdout_idx = holdout_fold.test_idx
        clf_report = classification_report_dict(y_full[holdout_idx], y_pred)
        mlflow.log_metrics(clf_report)
        mlflow.log_metrics(
            {
                "holdout_ece_raw": expected_calibration_error(y_full[holdout_idx], y_proba_raw),
                "holdout_ece_calibrated": expected_calibration_error(y_full[holdout_idx], y_proba),
            }
        )

        # See train_lightgbm.py's matching block for why the gate uses
        # compounded_report (single capital pool, sequentially
        # compounded) rather than financial_report (sums every signal's
        # return independently, overstating returns once trade windows
        # overlap — routine here, since a triple-barrier trade can stay
        # open for horizon_bars while a new signal fires every bar).
        returns = strategy_returns(close, holdout_idx, label_end_idx[holdout_idx], y_pred, LABEL_TO_INT, cost_bps)
        champion_fin = financial_report(returns)

        trade_returns = sequential_trade_returns(close, holdout_idx, label_end_idx[holdout_idx], y_pred, LABEL_TO_INT, cost_bps)
        champion_compounded = compounded_report(trade_returns)

        baseline_majority = majority_class_baseline(len(holdout_idx))
        baseline_bh = buy_and_hold_baseline(close, holdout_idx, cost_bps)

        log_metrics_safe(
            {
                "holdout_net_pnl": champion_fin.net_pnl,
                "holdout_sharpe": champion_fin.sharpe,
                "holdout_n_trades": champion_fin.n_trades,
                "holdout_win_rate": champion_fin.win_rate,
                "holdout_compounded_return": champion_compounded.total_return,
                "holdout_compounded_sharpe": champion_compounded.sharpe,
                "holdout_compounded_n_trades": champion_compounded.n_trades,
                "holdout_compounded_win_rate": champion_compounded.win_rate,
                "holdout_compounded_max_drawdown": champion_compounded.max_drawdown,
                "baseline_majority_net_pnl": baseline_majority.net_pnl,
                "baseline_buy_and_hold_net_pnl": baseline_bh.net_pnl,
            }
        )

        beats_baselines = (
            champion_compounded.total_return > baseline_majority.net_pnl and champion_compounded.total_return > baseline_bh.net_pnl
        )
        mlflow.log_param("beats_baselines_net_of_cost", beats_baselines)
        mlflow.log_param("calibration_method", calibration_method)

        scaled_full = scaler.transform(X_full).astype(np.float32)
        test_ds = SequenceWindowDataset(scaled_full, y_full, holdout_idx, seq_len)
        test_loader = _make_loader(test_ds, best["batch_size"], shuffle=False, device=device)
        mean_attn = _collect_mean_attention(champion, test_loader, device)

        with tempfile.TemporaryDirectory() as tmp:
            cm_path = os.path.join(tmp, "confusion_matrix.png")
            cal_raw_path = os.path.join(tmp, "calibration_curve_raw.png")
            cal_calibrated_path = os.path.join(tmp, "calibration_curve_calibrated.png")
            equity_path = os.path.join(tmp, "equity_curve.png")

            plot_confusion_matrix(y_full[holdout_idx], y_pred, cm_path)
            plot_calibration_curve(y_full[holdout_idx], y_proba_raw, cal_raw_path)
            plot_calibration_curve(y_full[holdout_idx], y_proba, cal_calibrated_path)
            plot_equity_curve(compounded_equity_curve(trade_returns), equity_path)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(cal_raw_path)
            mlflow.log_artifact(cal_calibrated_path)
            mlflow.log_artifact(equity_path)

            if mean_attn is not None:
                attn_path = os.path.join(tmp, "attention_weights.png")
                _plot_attention_weights(mean_attn, attn_path)
                mlflow.log_artifact(attn_path)

            import joblib

            scaler_path = os.path.join(tmp, "scaler.joblib")
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path)

            calibrators_path = os.path.join(tmp, "calibrators.joblib")
            joblib.dump(calibration_result, calibrators_path)
            mlflow.log_artifact(calibrators_path)

        # serialization_format="pickle" (not the newer default "pt2" traced
        # format): pt2 requires a strict TensorSpec signature and tracing
        # via the input example, which breaks when the model lives on
        # cuda but the example tensor is built on cpu — pickle just saves
        # the module directly, no device-bound tracing involved.
        input_example = np.zeros((1, seq_len, X_full.shape[1]), dtype=np.float32)
        model_info = mlflow.pytorch.log_model(champion, name="model", input_example=input_example, serialization_format="pickle")

        registered_name = f"{experiment_name}_{symbol}_{timeframe}"
        mv = mlflow.register_model(model_info.model_uri, registered_name)
        client = mlflow.MlflowClient()
        if beats_baselines:
            client.set_registered_model_alias(registered_name, "champion", mv.version)
            logger.info("Registered %s v%s as champion (beats baselines net of cost)", registered_name, mv.version)
        else:
            logger.warning(
                "%s v%s does NOT beat baselines net of cost — registered but not aliased 'champion'",
                registered_name,
                mv.version,
            )

        return parent_run.info.run_id

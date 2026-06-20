"""LightGBM training: Optuna hyperparameter search + MLflow tracking.

Implements docs/03's LightGBM section end to end:
- Optuna tunes hyperparameters; every trial is logged to MLflow as a
  nested run. The objective is evaluated under purged/embargoed
  walk-forward CV (ml_pipeline.validation) — never a leaky split.
- Early stopping uses a chronological validation slice carved from the
  tail of each fold's training set (the most recent bars before that
  fold's test set), not a random split.
- A final out-of-time holdout (never touched during tuning) gives the
  reported metrics and decides champion status.
- Metrics: macro-F1 / per-class (never lead with raw accuracy — Hold
  dominates), plus the financial reward metric (net PnL, Sharpe, net of
  transaction costs) via ml_pipeline.financial_metrics. A model must
  beat both required baselines (majority-Hold, buy-and-hold) net of
  costs to register as champion — beating F1 alone is not enough.
- Artifacts logged per run: model, confusion matrix, calibration curve,
  SHAP summary, equity curve.

Champion selection note: this targets mlflow>=2.10, where the legacy
Staging/Production *stages* API (named in docs/03) is deprecated in
favor of registered-model *aliases*. This module registers the model
and sets the "champion" alias only when it beats both baselines net of
cost on the holdout; otherwise it's left registered without an alias
(inspectable, not deployed) — the functional equivalent of "Staging".

ONNX export is intentionally not done here — it's the dedicated job of
the planned ml_pipeline/export_onnx.py, which exports whatever run holds
the "champion" alias.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import shap

from data_pipeline.labeling import INT_TO_LABEL, LABEL_TO_INT
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
    financial_report,
    majority_class_baseline,
    strategy_returns,
)
from ml_pipeline.validation import out_of_time_holdout_split, purged_walk_forward_splits
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _train_one_fold(
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    fit_idx, es_idx = carve_early_stopping_val(train_idx)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=NUM_CLASS,
        random_state=seed,
        verbosity=-1,
        **params,
    )
    model.fit(
        X.iloc[fit_idx],
        y[fit_idx],
        eval_set=[(X.iloc[es_idx], y[es_idx])],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    preds = model.predict(X.iloc[test_idx])
    return preds, model.predict_proba(X.iloc[test_idx])


@dataclass
class TuningContext:
    X: pd.DataFrame
    y: np.ndarray
    close: np.ndarray
    label_end_idx: np.ndarray
    folds: list
    seed: int
    cost_bps: float


def _objective(trial: optuna.Trial, ctx: TuningContext, objective_metric: str) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "class_weight": "balanced",  # docs/03: handle Hold-imbalance via class_weight, not oversampling
    }

    fold_scores = []
    with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_param("seed", ctx.seed)

        for fold_i, fold in enumerate(ctx.folds):
            preds, _ = _train_one_fold(params, ctx.X, ctx.y, fold.train_idx, fold.test_idx, ctx.seed)

            if objective_metric == "financial":
                returns = strategy_returns(
                    ctx.close, fold.test_idx, ctx.label_end_idx[fold.test_idx], preds, LABEL_TO_INT, ctx.cost_bps
                )
                score = financial_report(returns).net_pnl
            else:
                score = classification_report_dict(ctx.y[fold.test_idx], preds)["f1_macro"]

            mlflow.log_metric(f"fold{fold_i}_{objective_metric}", score)
            fold_scores.append(score)

        mean_score = float(np.nanmean(fold_scores))
        mlflow.log_metric(f"mean_{objective_metric}", mean_score)

    return mean_score


def _plot_shap_summary(model: lgb.LGBMClassifier, X_sample: pd.DataFrame, path: str) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    fig = plt.figure(figsize=(6, 5))
    shap.summary_plot(shap_values, X_sample, show=False, class_names=[INT_TO_LABEL[i] for i in (0, 1, 2)])
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
    n_trials: int = 30,
    embargo_bars: int = 10,
    holdout_frac: float = 0.15,
    cost_bps: float = DEFAULT_COST_BPS,
    objective_metric: str = "f1_macro",
    seed: int = 42,
    experiment_name: str = "lightgbm_buy_hold_sell",
    labeled_df: pd.DataFrame | None = None,
) -> str:
    """Full train+tune+evaluate+register pipeline for one (symbol, timeframe).

    `labeled_df` is a testing hook: pass a pre-built labeled DataFrame
    (same shape as build_labeled_dataset's output) to skip the live data
    fetch — used by the test suite to exercise this pipeline offline.

    Returns the MLflow run ID of the parent (champion) run.
    """
    set_global_seed(seed)
    mlflow.set_experiment(experiment_name)

    labeled = (
        labeled_df
        if labeled_df is not None
        else build_labeled_dataset(symbol, timeframe, source, lookback_days, horizon_bars, horizon_bucket)
    )
    if len(labeled) < 50:
        raise ValueError(f"Not enough labeled samples ({len(labeled)}) to train on")

    X = labeled[FEATURE_COLUMNS]
    # Unlabelable rows get a -1 placeholder; never selected for train/test
    # since label_end_idx==-1 excludes them in validation.py's valid_mask.
    y = labeled["label"].map(LABEL_TO_INT).fillna(-1).astype(int).to_numpy()
    close = labeled["close"].to_numpy()
    label_end_idx = labeled["label_end_idx"].to_numpy()
    n_total = len(labeled)

    holdout_fold = out_of_time_holdout_split(n_total, label_end_idx, holdout_frac, embargo_bars)
    n_tuning = int(holdout_fold.test_idx.min())

    folds = list(
        purged_walk_forward_splits(n_tuning, label_end_idx[:n_tuning], n_splits=n_splits, embargo_bars=embargo_bars)
    )
    if not folds:
        raise ValueError("No valid CV folds — increase lookback_days or reduce n_splits/embargo_bars")

    ctx = TuningContext(X=X, y=y, close=close, label_end_idx=label_end_idx, folds=folds, seed=seed, cost_bps=cost_bps)

    with mlflow.start_run(run_name=f"{symbol}_{timeframe}") as parent_run:
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
                "feature_set_hash": feature_set_hash(FEATURE_COLUMNS),
                "data_start": str(labeled.index.min()),
                "data_end": str(labeled.index.max()),
                "n_samples": n_total,
            }
        )

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(lambda trial: _objective(trial, ctx, objective_metric), n_trials=n_trials)

        best_params = dict(study.best_params)
        best_params["class_weight"] = "balanced"
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric(f"best_{objective_metric}", study.best_value)

        # Final champion: train on the full purged/embargoed tuning set,
        # evaluate once on the never-touched holdout.
        fit_idx, es_idx = carve_early_stopping_val(holdout_fold.train_idx)
        champion = lgb.LGBMClassifier(
            objective="multiclass", num_class=NUM_CLASS, random_state=seed, verbosity=-1, **best_params
        )
        champion.fit(
            X.iloc[fit_idx],
            y[fit_idx],
            eval_set=[(X.iloc[es_idx], y[es_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        holdout_idx = holdout_fold.test_idx
        y_pred = champion.predict(X.iloc[holdout_idx])
        y_proba = champion.predict_proba(X.iloc[holdout_idx])

        clf_report = classification_report_dict(y[holdout_idx], y_pred)
        mlflow.log_metrics(clf_report)

        returns = strategy_returns(close, holdout_idx, label_end_idx[holdout_idx], y_pred, LABEL_TO_INT, cost_bps)
        champion_fin = financial_report(returns)
        baseline_majority = majority_class_baseline(len(holdout_idx))
        baseline_bh = buy_and_hold_baseline(close, holdout_idx, cost_bps)

        log_metrics_safe(
            {
                "holdout_net_pnl": champion_fin.net_pnl,
                "holdout_sharpe": champion_fin.sharpe,
                "holdout_n_trades": champion_fin.n_trades,
                "holdout_win_rate": champion_fin.win_rate,
                "baseline_majority_net_pnl": baseline_majority.net_pnl,
                "baseline_buy_and_hold_net_pnl": baseline_bh.net_pnl,
            }
        )

        beats_baselines = (
            champion_fin.net_pnl > baseline_majority.net_pnl and champion_fin.net_pnl > baseline_bh.net_pnl
        )
        mlflow.log_param("beats_baselines_net_of_cost", beats_baselines)

        with tempfile.TemporaryDirectory() as tmp:
            cm_path = os.path.join(tmp, "confusion_matrix.png")
            cal_path = os.path.join(tmp, "calibration_curve.png")
            equity_path = os.path.join(tmp, "equity_curve.png")
            shap_path = os.path.join(tmp, "shap_summary.png")

            plot_confusion_matrix(y[holdout_idx], y_pred, cm_path)
            plot_calibration_curve(y[holdout_idx], y_proba, cal_path)
            plot_equity_curve(returns[returns != 0], equity_path)
            _plot_shap_summary(champion, X.iloc[holdout_idx], shap_path)

            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(cal_path)
            mlflow.log_artifact(equity_path)
            mlflow.log_artifact(shap_path)

        model_info = mlflow.lightgbm.log_model(champion, name="model")

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

"""Evaluation plots shared by train_lightgbm.py and train_lstm.py.

One canonical confusion-matrix/calibration/equity-curve implementation
so both models' MLflow artifacts are visually comparable apples-to-apples.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from data_pipeline.labeling import INT_TO_LABEL


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    ConfusionMatrixDisplay(cm, display_labels=[INT_TO_LABEL[i] for i in (0, 1, 2)]).plot(ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration_curve(y_true: np.ndarray, proba: np.ndarray, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for class_id, name in INT_TO_LABEL.items():
        binary_true = (y_true == class_id).astype(int)
        bin_edges = np.linspace(0, 1, 11)
        bin_idx = np.clip(np.digitize(proba[:, class_id], bin_edges) - 1, 0, len(bin_edges) - 2)
        observed = [binary_true[bin_idx == b].mean() if (bin_idx == b).any() else np.nan for b in range(len(bin_edges) - 1)]
        predicted = [proba[bin_idx == b, class_id].mean() if (bin_idx == b).any() else np.nan for b in range(len(bin_edges) - 1)]
        ax.plot(predicted, observed, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_equity_curve(returns: np.ndarray, path: str) -> None:
    equity = np.cumsum(returns)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(equity)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative return")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

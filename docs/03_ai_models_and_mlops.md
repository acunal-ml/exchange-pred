# AI Architecture and MLOps

## Validation Methodology (the single most important fix — was entirely missing)

Financial time series **must not** use random `KFold`, shuffling, or any default `train_test_split`. Doing so leaks the future into the past and produces backtests that look excellent and fail live. This is the #1 way these projects die.

- Use **walk-forward** evaluation or **purged + embargoed time-series cross-validation** (López de Prado style). Splits respect chronological order.
- **Purge** training samples whose label window overlaps the test window; apply an **embargo** gap after each test fold to kill serial-correlation leakage.
- Maintain a final **out-of-time holdout** never touched during tuning — this is what feeds the UI backtest panel.
- Fix all random seeds; log them.

## MLOps Framework (MLflow)
- All training runs tracked in `mlflow`, including hyperparameters, seeds, data version/date range, and feature set hash.
- **Classification metrics:** Precision, Recall, **F1 (macro)**, and per-class metrics. **Do not lead with raw Accuracy** — with `Hold` dominating, accuracy is misleading.
- **Financial Reward metric (now defined, previously undefined):** simulated out-of-sample **net PnL** and **Sharpe ratio** of acting on the signals over the validation window, **after** transaction costs and slippage. A model that wins on F1 but loses money is not the champion.
- **Artifacts logged per run:** trained model (`.joblib`/native for LightGBM, `.pt` for PyTorch), **ONNX export**, confusion matrix, calibration curve, SHAP feature-importance plot, and the equity curve.
- **Champion selection** uses the **MLflow Model Registry** (Staging → Production stages), not just a loose `mlruns/` folder. Only Production-stage models are exported for deployment.
- Always log a **baseline** to beat: majority-class (`always Hold`) and buy-and-hold. If the model can't beat these net of costs, it doesn't ship.

## Probability Calibration (required for the UI threshold slider)
- Raw LightGBM/LSTM probabilities are usually over/under-confident. Calibrate with **isotonic or Platt scaling** on a held-out fold so the Tab-2 confidence threshold has real meaning.
- Class imbalance is handled via `class_weight`/`scale_pos_weight` (LightGBM) or weighted/focal loss (LSTM) — **not** by naive oversampling that distorts the live distribution.

---

## 1. Machine Learning: LightGBM
- **Purpose:** high-performance tabular classification (Buy / Hold / Sell).
- **Optimization:** hyperparameter tuning via **Optuna**, each trial logged to MLflow; objective is the financial reward metric (or macro-F1), evaluated under purged CV — never a leaky split.
- Use early stopping on a chronological validation fold; cap tree depth/leaves to limit overfitting on noisy financial features.

## 2. Deep Learning: Attention-based LSTM
- **Architecture:** stacked LSTM with an attention mechanism weighting critical historical time steps; dropout for regularization.
- **Hardware:** developed on an RTX 3050 (**4GB VRAM**).
- **Resource management (must):**
  - Optimized **mini-batching** + **gradient accumulation** to fit 4GB and avoid CUDA OOM.
  - **Mixed precision (fp16/AMP)**.
  - **Gradient clipping** (LSTMs are prone to exploding gradients).
  - Keep **sequence length** and batch size tuned to the VRAM budget; prefer shorter windows + accumulation over a single large batch.
  - Use a streaming/windowed `DataLoader` (`num_workers`, `pin_memory`) so full tensors aren't materialized in RAM.
- **Reproducibility:** deterministic seeds, logged sequence length, and saved scaler/normalizer as an MLflow artifact (the same scaler must be applied at inference — train/serve skew here is a classic silent bug).

## Train → Serve handoff
- The champion is exported to **ONNX** (and optionally **int8-quantized**) and pushed to an HF Hub / Dataset repo. The HF Space only loads and runs this artifact — it never trains. See deployment doc.

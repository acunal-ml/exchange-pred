"""Cost-aware financial reward metrics and baselines.

docs/03: "A model that wins on F1 but loses money is not the champion."
Classification metrics alone (precision/recall/F1) say nothing about
whether acting on a model's signals is profitable after costs. This
module simulates net PnL by reusing each sample's existing
triple-barrier outcome window — entry at index i, exit at
label_end_idx[i] (the barrier touch, or horizon end for Hold) — so the
"trade" is exactly the event the label was built from.

Always evaluate alongside the two required baselines (docs/03):
majority-class ("always Hold") and buy-and-hold. A champion must beat
both, net of costs, or it doesn't ship.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Round-trip transaction cost + slippage, in basis points. A Buy or Sell
# trade pays this; Hold (no position) pays nothing.
DEFAULT_COST_BPS = 15.0


@dataclass(frozen=True)
class FinancialReport:
    net_pnl: float
    sharpe: float
    n_trades: int
    win_rate: float


def _direction(label_ints: np.ndarray, label_to_int: dict[str, int]) -> np.ndarray:
    """Map encoded labels to trade direction: Buy=+1, Sell=-1, Hold=0."""
    direction = np.zeros(len(label_ints), dtype=float)
    direction[label_ints == label_to_int["Buy"]] = 1.0
    direction[label_ints == label_to_int["Sell"]] = -1.0
    return direction


def strategy_returns(
    close: np.ndarray,
    entry_idx: np.ndarray,
    exit_idx: np.ndarray,
    label_ints: np.ndarray,
    label_to_int: dict[str, int],
    cost_bps: float = DEFAULT_COST_BPS,
) -> np.ndarray:
    """Per-sample realized return from acting on each predicted label.

    Hold positions return 0 (no trade, no cost). Buy/Sell positions earn
    the directional price move from entry to exit, minus the round-trip
    cost in bps.
    """
    direction = _direction(label_ints, label_to_int)
    raw_return = direction * (close[exit_idx] / close[entry_idx] - 1.0)
    cost = np.where(direction != 0, cost_bps / 10_000.0, 0.0)
    return raw_return - cost


def financial_report(returns: np.ndarray) -> FinancialReport:
    trades = returns[returns != 0]
    if len(trades) == 0:
        return FinancialReport(net_pnl=0.0, sharpe=float("nan"), n_trades=0, win_rate=float("nan"))

    net_pnl = float(trades.sum())
    std = trades.std(ddof=0)
    # Trade-level Sharpe (not annualized — trade frequency varies by
    # timeframe/horizon, so annualizing would need a periods-per-year
    # assumption that doesn't belong in a generic metric helper).
    sharpe = float(trades.mean() / std) if std > 0 else float("nan")
    win_rate = float((trades > 0).mean())
    return FinancialReport(net_pnl=net_pnl, sharpe=sharpe, n_trades=len(trades), win_rate=win_rate)


def majority_class_baseline(n_samples: int) -> FinancialReport:
    """Always predict Hold -> never trades, zero PnL, zero risk."""
    return financial_report(np.zeros(n_samples))


def buy_and_hold_baseline(close: np.ndarray, entry_idx: np.ndarray, cost_bps: float = DEFAULT_COST_BPS) -> FinancialReport:
    """Buy once at the first sample, hold through the last sample's close."""
    if len(entry_idx) == 0:
        return financial_report(np.array([]))
    first, last = close[entry_idx[0]], close[entry_idx[-1]]
    ret = (last / first - 1.0) - cost_bps / 10_000.0
    return financial_report(np.array([ret]))

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
    """Sums every signal's return independently.

    Caveat (read before trusting the headline number): this treats every
    triggered signal as if it had its own fully-funded, independent
    capital allocation. Since a triple-barrier trade can stay open for
    `horizon_bars` bars but a new signal is evaluated every bar, trade
    windows routinely overlap — summing them overstates what a single
    capital pool could actually realize. It's a fair *relative*
    comparison against a baseline computed the same way (which is what
    the champion-gating check in train_lightgbm.py/train_lstm.py uses
    it for), but not a literal "your capital grows by this much" number.
    For that, use `sequential_trade_returns` + `compounded_report`
    below, which enforce a single open position at a time.
    """
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


@dataclass(frozen=True)
class CompoundedReport:
    total_return: float  # compounded return over the whole sequence, e.g. 0.20 = +20%
    sharpe: float
    n_trades: int
    win_rate: float
    max_drawdown: float  # most negative peak-to-trough move of the compounded equity curve
    equity_curve: list[float]  # cumulative compounded return path, one point per trade


@dataclass(frozen=True)
class Trade:
    entry_idx: int
    exit_idx: int
    direction: str  # "Buy" | "Sell" (Hold never produces a Trade record)
    ret: float


def sequential_trades(
    close: np.ndarray,
    entry_idx: np.ndarray,
    exit_idx: np.ndarray,
    label_ints: np.ndarray,
    label_to_int: dict[str, int],
    cost_bps: float = DEFAULT_COST_BPS,
) -> list[Trade]:
    """Like strategy_returns, but enforces a single capital pool: at most
    one open position at a time. A signal that fires while a previous
    trade's barrier hasn't resolved yet is skipped entirely — capital is
    committed until `exit_idx` of the open trade. This is what makes the
    resulting returns compoundable (cumulative-product) into a real
    equity curve, unlike strategy_returns' independent per-signal sums.

    Returns structured `Trade` records (which position opened/closed it,
    direction, realized return) rather than bare floats — needed
    wherever the UI shows *which* trades these were (e.g. a recent-
    trades table), not just aggregate stats. Use `sequential_trade_returns`
    below if all you need is the bare return array.

    `entry_idx`/`exit_idx`/`label_ints` are parallel arrays (same
    convention as strategy_returns — `entry_idx` need not be sorted,
    this function sorts internally to walk chronologically).
    """
    int_to_label = {v: k for k, v in label_to_int.items()}
    direction_map = {label_to_int["Buy"]: 1.0, label_to_int["Sell"]: -1.0}
    order = np.argsort(entry_idx)

    trades = []
    next_available = -1
    for k in order:
        pos = entry_idx[k]
        if pos < next_available:
            continue
        direction = direction_map.get(label_ints[k])
        if direction is None:
            continue
        exit_pos = exit_idx[k]
        ret = direction * (close[exit_pos] / close[pos] - 1.0) - cost_bps / 10_000.0
        trades.append(Trade(entry_idx=int(pos), exit_idx=int(exit_pos), direction=int_to_label[label_ints[k]], ret=float(ret)))
        next_available = exit_pos + 1
    return trades


def sequential_trade_returns(
    close: np.ndarray,
    entry_idx: np.ndarray,
    exit_idx: np.ndarray,
    label_ints: np.ndarray,
    label_to_int: dict[str, int],
    cost_bps: float = DEFAULT_COST_BPS,
) -> np.ndarray:
    """Thin wrapper over sequential_trades() for callers (e.g. the
    champion-gating check) that only need the bare return values, not
    which trades produced them."""
    trades = sequential_trades(close, entry_idx, exit_idx, label_ints, label_to_int, cost_bps)
    return np.array([t.ret for t in trades])


def compounded_equity_curve(trade_returns: np.ndarray) -> np.ndarray:
    """Cumulative compounded return path: e.g. [0.05, -0.02, 0.03] ->
    [0.05, 0.029, 0.0599] (each point is total return-to-date, not the
    per-trade return)."""
    if len(trade_returns) == 0:
        return np.array([])
    return np.cumprod(1.0 + trade_returns) - 1.0


def compounded_report(trade_returns: np.ndarray) -> CompoundedReport:
    """The realistic counterpart to financial_report(): one capital pool,
    sequentially compounded, true max drawdown. Use this (not
    financial_report) whenever the number will be shown to a user as
    "what would this strategy have returned.\""""
    if len(trade_returns) == 0:
        return CompoundedReport(total_return=0.0, sharpe=float("nan"), n_trades=0, win_rate=float("nan"), max_drawdown=0.0, equity_curve=[])

    equity = compounded_equity_curve(trade_returns)
    running_max = np.maximum.accumulate(1.0 + equity)
    drawdown = (1.0 + equity) / running_max - 1.0

    std = trade_returns.std(ddof=0)
    sharpe = float(trade_returns.mean() / std) if std > 0 else float("nan")
    win_rate = float((trade_returns > 0).mean())

    return CompoundedReport(
        total_return=float(equity[-1]),
        sharpe=sharpe,
        n_trades=len(trade_returns),
        win_rate=win_rate,
        max_drawdown=float(drawdown.min()),
        equity_curve=equity.tolist(),
    )


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

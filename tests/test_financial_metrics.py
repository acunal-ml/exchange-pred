import numpy as np

from data_pipeline.labeling import LABEL_TO_INT
from ml_pipeline.financial_metrics import (
    buy_and_hold_baseline,
    compounded_equity_curve,
    compounded_report,
    financial_report,
    majority_class_baseline,
    sequential_trade_returns,
    strategy_returns,
)


def test_buy_trade_profits_net_of_cost():
    close = np.array([100.0, 110.0])
    entry_idx = np.array([0])
    exit_idx = np.array([1])
    label_ints = np.array([LABEL_TO_INT["Buy"]])
    returns = strategy_returns(close, entry_idx, exit_idx, label_ints, LABEL_TO_INT, cost_bps=10.0)
    expected = (110 / 100 - 1) - 0.001
    assert np.isclose(returns[0], expected)


def test_sell_trade_profits_on_price_drop():
    close = np.array([100.0, 90.0])
    entry_idx = np.array([0])
    exit_idx = np.array([1])
    label_ints = np.array([LABEL_TO_INT["Sell"]])
    returns = strategy_returns(close, entry_idx, exit_idx, label_ints, LABEL_TO_INT, cost_bps=0.0)
    assert returns[0] > 0


def test_hold_has_zero_return_and_zero_cost():
    close = np.array([100.0, 90.0])
    entry_idx = np.array([0])
    exit_idx = np.array([1])
    label_ints = np.array([LABEL_TO_INT["Hold"]])
    returns = strategy_returns(close, entry_idx, exit_idx, label_ints, LABEL_TO_INT, cost_bps=50.0)
    assert returns[0] == 0.0


def test_majority_class_baseline_is_zero_pnl():
    report = majority_class_baseline(50)
    assert report.net_pnl == 0.0
    assert report.n_trades == 0


def test_buy_and_hold_baseline_matches_total_return():
    close = np.array([100.0, 105.0, 120.0])
    entry_idx = np.array([0, 1, 2])
    report = buy_and_hold_baseline(close, entry_idx, cost_bps=0.0)
    assert np.isclose(report.net_pnl, 120 / 100 - 1)


def test_financial_report_win_rate():
    returns = np.array([0.05, -0.02, 0.0, 0.03, -0.01])
    report = financial_report(returns)
    assert report.n_trades == 4  # zeros excluded (no trade)
    assert np.isclose(report.win_rate, 2 / 4)


def test_sequential_trade_returns_skips_signals_while_a_position_is_open():
    # A trade opened at position 0 stays open until exit_idx=3. A second
    # signal fires at position 1 (entry_idx) while that's still open —
    # it must be skipped entirely, unlike strategy_returns which would
    # happily count both as independent trades.
    close = np.array([100.0, 110.0, 105.0, 102.0, 130.0, 90.0])
    entry_idx = np.array([0, 1, 4])
    exit_idx = np.array([3, 2, 5])
    label_ints = np.array([LABEL_TO_INT["Buy"], LABEL_TO_INT["Buy"], LABEL_TO_INT["Sell"]])

    trades = sequential_trade_returns(close, entry_idx, exit_idx, label_ints, LABEL_TO_INT, cost_bps=0.0)

    assert len(trades) == 2  # the position-1 signal was skipped
    assert np.isclose(trades[0], 102.0 / 100.0 - 1.0)
    assert np.isclose(trades[1], -(90.0 / 130.0 - 1.0))


def test_sequential_trade_returns_sorts_unordered_entry_idx():
    close = np.array([100.0, 200.0, 105.0, 110.0])
    entry_idx = np.array([2, 0])  # deliberately out of chronological order
    exit_idx = np.array([3, 1])
    label_ints = np.array([LABEL_TO_INT["Buy"], LABEL_TO_INT["Buy"]])

    trades = sequential_trade_returns(close, entry_idx, exit_idx, label_ints, LABEL_TO_INT, cost_bps=0.0)
    # processed in time order: position 0 first (entry=100, exit=200), then position 2 (entry=105, exit=110)
    assert len(trades) == 2
    assert np.isclose(trades[0], 200.0 / 100.0 - 1.0)
    assert np.isclose(trades[1], 110.0 / 105.0 - 1.0)


def test_compounded_equity_curve_differs_from_naive_sum():
    # +10% then -10% nets to -1% compounded, not 0% — the whole point of
    # this function versus a plain additive cumsum.
    trade_returns = np.array([0.10, -0.10])
    equity = compounded_equity_curve(trade_returns)
    assert np.isclose(equity[-1], -0.01)
    assert not np.isclose(equity[-1], trade_returns.sum())


def test_compounded_report_max_drawdown_is_negative_after_a_loss():
    trade_returns = np.array([0.10, -0.20, 0.05])
    report = compounded_report(trade_returns)
    assert report.max_drawdown < 0.0
    assert report.n_trades == 3
    assert len(report.equity_curve) == 3


def test_compounded_report_empty_is_zero_not_nan_net():
    report = compounded_report(np.array([]))
    assert report.total_return == 0.0
    assert report.n_trades == 0
    assert report.equity_curve == []

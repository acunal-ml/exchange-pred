import numpy as np

from data_pipeline.labeling import LABEL_TO_INT
from ml_pipeline.financial_metrics import (
    buy_and_hold_baseline,
    financial_report,
    majority_class_baseline,
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

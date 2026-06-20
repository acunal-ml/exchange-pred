import numpy as np
import pytest

from inference.signal_aggregator import (
    SELL,
    HOLD,
    BUY,
    bollinger_vote,
    combine_indicator_votes,
    compute_levels,
    fuse_signals,
    macd_vote,
    rsi_vote,
)


def test_rsi_vote_oversold_leans_buy():
    vote = rsi_vote(20)
    assert vote[BUY] > vote[SELL]
    assert np.isclose(vote.sum(), 1.0)


def test_rsi_vote_overbought_leans_sell():
    vote = rsi_vote(80)
    assert vote[SELL] > vote[BUY]


def test_rsi_vote_neutral_leans_hold():
    vote = rsi_vote(50)
    assert vote[HOLD] >= vote[BUY]
    assert vote[HOLD] >= vote[SELL]


def test_macd_vote_positive_histogram_leans_buy():
    vote = macd_vote(macd_hist=2.0, atr=1.0)
    assert vote[BUY] > vote[SELL]


def test_macd_vote_zero_atr_does_not_crash():
    vote = macd_vote(macd_hist=2.0, atr=0.0)
    assert np.isclose(vote.sum(), 1.0)


def test_bollinger_vote_below_lower_band_leans_buy():
    vote = bollinger_vote(percent_b=-0.1)
    assert vote[BUY] > vote[SELL]


def test_bollinger_vote_above_upper_band_leans_sell():
    vote = bollinger_vote(percent_b=1.2)
    assert vote[SELL] > vote[BUY]


def test_combine_indicator_votes_normalizes_weights():
    votes_equal = combine_indicator_votes(rsi=20, macd_hist=2.0, atr=1.0, percent_b=-0.1)
    votes_weighted = combine_indicator_votes(
        rsi=20, macd_hist=2.0, atr=1.0, percent_b=-0.1, indicator_weights={"rsi": 10, "macd": 0, "bollinger": 0}
    )
    assert np.isclose(votes_equal.sum(), 1.0)
    assert np.allclose(votes_weighted, rsi_vote(20))


def test_fuse_signals_label_follows_argmax_above_threshold():
    p_ind = np.array([0.1, 0.1, 0.8])
    result = fuse_signals(
        P_indicators=p_ind, P_lgbm=None, P_lstm=None,
        w_ind=1.0, w_lgbm=0.0, w_lstm=0.0,
        confidence_threshold=0.4, timeframe="1D", close=100.0, atr=2.0,
    )
    assert result.label == "Buy"
    assert np.isclose(result.confidence, 0.8)


def test_fuse_signals_falls_back_to_hold_below_threshold():
    p_ind = np.array([0.2, 0.4, 0.4])
    result = fuse_signals(
        P_indicators=p_ind, P_lgbm=None, P_lstm=None,
        w_ind=1.0, w_lgbm=0.0, w_lstm=0.0,
        confidence_threshold=0.6, timeframe="1D", close=100.0, atr=2.0,
    )
    assert result.label == "Hold"


def test_fuse_signals_excludes_none_sources_from_weighting():
    p_ind = np.array([0.0, 0.0, 1.0])
    p_lgbm = np.array([1.0, 0.0, 0.0])
    # lgbm weight is 0 -> must NOT affect the fused result despite strong Sell signal
    result = fuse_signals(
        P_indicators=p_ind, P_lgbm=p_lgbm, P_lstm=None,
        w_ind=1.0, w_lgbm=0.0, w_lstm=0.0,
        confidence_threshold=0.5, timeframe="1D", close=100.0, atr=2.0,
    )
    assert result.label == "Buy"


def test_fuse_signals_requires_at_least_one_active_source():
    with pytest.raises(ValueError):
        fuse_signals(
            P_indicators=None, P_lgbm=None, P_lstm=None,
            w_ind=0.0, w_lgbm=0.0, w_lstm=0.0,
            confidence_threshold=0.5, timeframe="1D", close=100.0, atr=2.0,
        )


def test_compute_levels_buy_has_target_above_and_stop_below_entry():
    levels = compute_levels("Buy", close=100.0, atr=2.0, horizon_bucket="medium")
    assert levels["entry"] == 100.0
    assert levels["target"] > levels["entry"]
    assert levels["stop"] < levels["entry"]


def test_compute_levels_sell_has_target_below_and_stop_above_entry():
    levels = compute_levels("Sell", close=100.0, atr=2.0, horizon_bucket="medium")
    assert levels["target"] < levels["entry"]
    assert levels["stop"] > levels["entry"]


def test_compute_levels_hold_has_no_target_or_stop():
    levels = compute_levels("Hold", close=100.0, atr=2.0, horizon_bucket="medium")
    assert levels["target"] is None
    assert levels["stop"] is None

"""Module 2: signals, backtest engine, metrics, no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energycurves import curves
from energycurves.carry import backtest as bt
from energycurves.carry.metrics import max_drawdown, performance
from energycurves.carry.signals import deseasonalize, raw_carry
from energycurves.data.stitch import slot_returns


def _loop_reference(
    returns: pd.DataFrame, rolled: pd.DataFrame, rebal: pd.DataFrame, cost_bps: float
) -> pd.Series:
    """Independent day-by-day implementation of the backtest accounting."""
    w = pd.Series(0.0, index=returns.columns)
    out = []
    for day in returns.index:
        r = returns.loc[day].fillna(0.0)
        pnl = float((w * r).sum())  # yesterday's weights earn today's return
        traded = 0.0
        if day in rebal.index:
            new_w = rebal.loc[day]
            traded += float((new_w - w).abs().sum())
            w = new_w.copy()
        traded += float((2.0 * w.abs() * rolled.loc[day].astype(float)).sum())
        out.append(pnl - traded * cost_bps / 1e4)
    return pd.Series(out, index=returns.index)


def test_engine_matches_loop_reference() -> None:
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2020-01-01", periods=300)
    cols = ["A", "B", "C"]
    returns = pd.DataFrame(rng.normal(0, 0.01, (300, 3)), index=days, columns=cols)
    returns.iloc[5, 1] = np.nan  # a missing price
    rolled = pd.DataFrame(rng.random((300, 3)) < 0.05, index=days, columns=cols)
    rebal_days = days[::21]
    rebal = pd.DataFrame(rng.normal(0, 0.3, (len(rebal_days), 3)), index=rebal_days, columns=cols)
    got = bt.run_backtest(returns, rolled, rebal, cost_bps=7.0).daily["net"]
    expected = _loop_reference(returns, rolled, rebal, 7.0)
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), atol=1e-15)


def test_xs_weights_are_dollar_neutral_with_unit_gross() -> None:
    sig = pd.DataFrame([[0.3, -0.1, 0.05, -0.4]], columns=list("ABCD"))
    vol = pd.DataFrame([[0.3, 0.5, 0.4, 0.2]], columns=list("ABCD"))
    w = bt.target_weights(sig, vol, "XS", bt.CarryConfig()).iloc[0]
    assert np.isclose(w.sum(), 0.0)
    assert np.isclose(w.abs().sum(), 1.0)
    assert w["A"] > 0 and w["C"] > 0 and w["B"] < 0 and w["D"] < 0
    # inverse-vol within the long leg: C (vol 0.4) gets less than A (vol 0.3)
    assert w["A"] > w["C"]


def test_ts_and_long_only_weights() -> None:
    sig = pd.DataFrame([[0.3, -0.1, np.nan]], columns=list("ABC"))
    vol = pd.DataFrame([[0.2, 0.4, 0.3]], columns=list("ABC"))
    cfg = bt.CarryConfig(target_vol=0.1)
    ts = bt.target_weights(sig, vol, "TS", cfg).iloc[0]
    lo = bt.target_weights(sig, vol, "LO", cfg).iloc[0]
    assert np.allclose(ts.to_numpy(), [0.25, -0.125, 0.0])  # (0.1/vol)/N, N = 2 valid
    assert np.allclose(lo.to_numpy(), [0.25, 0.125, 0.0])


def test_rebalance_dates_are_nth_trading_day() -> None:
    days = pd.bdate_range("2021-03-01", "2021-05-31")
    reb = bt.rebalance_dates(days, 10)
    assert list(reb.strftime("%Y-%m-%d")) == ["2021-03-12", "2021-04-14", "2021-05-14"]


def test_deseasonalize_removes_shape_keeps_level_and_uses_past_only() -> None:
    dates = pd.bdate_range("2000-01-03", "2009-12-31")
    month = dates.month
    seasonal = np.where(np.isin(month, [11, 12, 1, 2]), 0.3, -0.1)
    sig = pd.DataFrame(
        {"date": dates, "code": "XX", "m1": dates.to_period("M").to_timestamp(), "raw": seasonal}
    )
    out = deseasonalize(sig)
    late = out[out["date"] >= "2005-01-01"]
    level = sig["raw"].mean()
    assert np.allclose(late["deseason"], level, atol=0.02)  # shape gone, level kept
    first_valid = out.dropna(subset=["deseason"])["date"].min()
    assert first_valid.year == 2003  # three full past years required
    # perturb the future: past values unchanged
    sig2 = sig.copy()
    sig2.loc[sig2["date"] >= "2007-01-01", "raw"] += 5.0
    out2 = deseasonalize(sig2)
    before = out["date"] < "2007-01-01"
    pd.testing.assert_series_equal(out.loc[before, "deseason"], out2.loc[before, "deseason"])


def test_no_look_ahead_in_carry_weights(panel: pd.DataFrame) -> None:
    """Scrambling every price after T leaves all weights decided up to T unchanged."""
    cut = pd.Timestamp("2015-06-30")
    rng = np.random.default_rng(0)
    noisy = panel.copy()
    after = noisy["date"] > cut
    noisy.loc[after, "settle"] *= np.exp(rng.normal(0, 0.2, after.sum()))
    cfg = bt.CarryConfig()

    def weights(p: pd.DataFrame, strategy: str, kind: str) -> pd.DataFrame:
        sig = deseasonalize(raw_carry(curves.roll_yields(p)))
        inp = bt.prepare_inputs(slot_returns(p, 2), sig, cfg)
        return bt.strategy_weights(inp, strategy, kind, "2010-01-01", "2016-12-31", config=cfg)

    for strategy, kind in (("TS", "deseason"), ("XS", "deseason"), ("TS", "raw"), ("LO", "raw")):
        w0, w1 = weights(panel, strategy, kind), weights(noisy, strategy, kind)
        upto = w0.index <= cut
        pd.testing.assert_frame_equal(w0[upto], w1[upto], check_exact=True)
        assert not w0[~upto].equals(w1[~upto])  # the perturbation did bite


@pytest.fixture(scope="module")
def carry_inputs(panel: pd.DataFrame) -> bt.CarryInputs:
    sig = deseasonalize(raw_carry(curves.roll_yields(panel)))
    return bt.prepare_inputs(slot_returns(panel, 2), sig, bt.CarryConfig())


def test_pnl_and_sharpe_decrease_with_costs(carry_inputs: bt.CarryInputs) -> None:
    metrics, sens, _ = bt.evaluate(carry_inputs, "2008-10-01", "2024-04-06", bt.CarryConfig())
    for _, grp in sens.groupby("strategy"):
        sharpes = grp.sort_values("cost_bps")["sharpe"].to_numpy()
        assert (np.diff(sharpes) < 0).all()
    assert set(metrics["strategy"]) == {"TS-raw", "TS-deseason", "XS-raw", "XS-deseason", "LO"}
    assert (metrics["start"] == metrics["start"].iloc[0]).all()  # same window for all


def test_metrics_on_known_paths() -> None:
    idx = pd.bdate_range("2020-01-01", periods=252)
    flat = pd.Series(0.001, index=idx)
    perf = performance(flat + np.tile([0.0005, -0.0005], 126))
    assert np.isclose(perf["ann_return"], 0.252)
    assert perf["max_drawdown"] <= 0
    path = pd.Series([0.1, -0.5, 0.2], index=idx[:3])
    assert np.isclose(max_drawdown(path), -0.5)

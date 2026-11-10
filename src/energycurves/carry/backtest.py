"""Carry backtests on the four-contract energy universe (module 2).

Timeline (no look-ahead): on each rebalance day ``t_r`` (the ``n``-th trading
day of the month, away from every expiry) the signal and the volatility are
measured at the close of ``t_r``; the new weights are traded at that close and
earn the returns of ``t_r + 1`` onwards. Every position is held in contract 2
and rolled when contract 1 expires (``energycurves.data.stitch.slot_returns``).

Strategies (Koijen, Moskowitz, Pedersen & Vrugt, 2018; Erb & Harvey, 2006):

* ``TS`` time series:  w_i = sign(s_i) · (σ* / σ̂_i) / N
* ``XS`` cross section: long the 2 highest, short the 2 lowest signals,
  inverse-volatility weights within each leg, legs of ±0.5 (gross 1, net 0)
* ``LO`` long only, equal risk: w_i = (σ* / σ̂_i) / N   (passive benchmark)

``σ̂_i`` is an exponentially weighted volatility of daily returns (span 60
days, annualised), known at the close of ``t_r``; σ* = 10 % a year.

Costs: ``c`` (in basis points of notional) on every unit of notional traded:
rebalancing trades ``|w_new − w_old|`` and each roll trades ``2 |w|`` (sell the
expiring-side contract, buy the next one).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energycurves.carry.metrics import performance


@dataclass(frozen=True)
class CarryConfig:
    """Backtest parameters (defaults used for the published results)."""

    rebalance_day: int = 10
    vol_span: int = 60
    target_vol: float = 0.10
    n_long: int = 2
    n_short: int = 2


def rebalance_dates(trading_days: pd.DatetimeIndex, nth: int) -> pd.DatetimeIndex:
    """The ``nth`` trading day of every calendar month present in ``trading_days``."""
    s = pd.Series(trading_days, index=trading_days)
    picked = s.groupby(trading_days.to_period("M")).nth(nth - 1)
    return pd.DatetimeIndex(picked.to_numpy())


def ewma_vol(returns: pd.DataFrame, span: int) -> pd.DataFrame:
    """Annualised EWMA volatility, using returns up to and including each date."""
    var = (returns.fillna(0.0) ** 2).ewm(span=span, min_periods=span).mean()
    vol: pd.DataFrame = (var * 252.0) ** 0.5
    return vol


def target_weights(
    signal: pd.DataFrame, vol: pd.DataFrame, strategy: str, config: CarryConfig
) -> pd.DataFrame:
    """Weights per rebalance date (rows) and asset (columns) for one strategy.

    ``signal`` and ``vol`` are already sampled on the rebalance dates; assets
    with a missing signal or volatility get a zero weight.
    """
    valid = signal.notna() & vol.notna() & (vol > 0)
    inv_vol = (config.target_vol / vol).where(valid)
    n_valid = valid.sum(axis=1).replace(0, np.nan)
    if strategy == "TS":
        sign = signal.gt(0).astype(float) - signal.lt(0).astype(float)
        ts: pd.DataFrame = (sign * inv_vol).div(n_valid, axis=0).fillna(0.0)
        return ts
    if strategy == "LO":
        return inv_vol.div(n_valid, axis=0).fillna(0.0)
    if strategy == "XS":
        rank = signal.where(valid).rank(axis=1, ascending=False, method="first")
        n = valid.sum(axis=1)
        long_leg = rank.le(config.n_long)
        short_leg = rank.gt(n - config.n_short, axis=0) & rank.notna()
        enough = n >= config.n_long + config.n_short
        lw = inv_vol.where(long_leg)
        sw = inv_vol.where(short_leg)
        w = (
            lw.div(lw.sum(axis=1), axis=0).fillna(0.0) * 0.5
            - sw.div(sw.sum(axis=1), axis=0).fillna(0.0) * 0.5
        )
        return w.where(enough, 0.0)
    raise ValueError(f"unknown strategy {strategy!r}")


@dataclass(frozen=True)
class BacktestResult:
    """Daily gross/net returns, traded notional and weights of one strategy."""

    daily: pd.DataFrame  # columns: gross, cost, net, traded
    weights: pd.DataFrame  # weights in force at each close (after rebalancing)


def run_backtest(
    returns: pd.DataFrame,
    rolled: pd.DataFrame,
    rebal_weights: pd.DataFrame,
    cost_bps: float = 0.0,
) -> BacktestResult:
    """Apply rebalance-date weights to daily asset returns, with trading costs.

    ``returns`` and ``rolled`` are daily (dates × assets); ``rebal_weights``
    holds the target weights decided at the close of each rebalance date.
    Missing asset returns count as zero.
    """
    dates = returns.index
    held = rebal_weights.reindex(dates).ffill().fillna(0.0)  # weights after today's close
    applied = held.shift(1).fillna(0.0)  # weights earning today's return
    gross = (applied * returns.fillna(0.0)).sum(axis=1)
    rebalanced = dates.isin(rebal_weights.index)
    trade_rebal = (held - applied).abs().sum(axis=1).where(rebalanced, 0.0)
    trade_roll = (2.0 * held.abs() * rolled.reindex(dates).fillna(False).astype(float)).sum(axis=1)
    traded = trade_rebal + trade_roll
    cost = traded * cost_bps / 1e4
    daily = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "traded": traded})
    return BacktestResult(daily=daily, weights=held)


@dataclass(frozen=True)
class CarryInputs:
    """Daily asset returns, roll flags, signals and volatilities (dates × assets)."""

    returns: pd.DataFrame
    rolled: pd.DataFrame
    raw: pd.DataFrame
    deseason: pd.DataFrame
    vol: pd.DataFrame


def prepare_inputs(slot2: pd.DataFrame, signals: pd.DataFrame, config: CarryConfig) -> CarryInputs:
    """Pivot slot-2 returns and signals into aligned (dates × assets) frames."""
    returns = slot2.pivot(index="date", columns="code", values="ret").sort_index()
    rolled = slot2.pivot(index="date", columns="code", values="rolled").reindex(returns.index)
    raw = signals.pivot(index="date", columns="code", values="raw").reindex(returns.index)
    deseason = signals.pivot(index="date", columns="code", values="deseason").reindex(returns.index)
    vol = ewma_vol(returns, config.vol_span)
    return CarryInputs(
        returns=returns,
        rolled=rolled.fillna(False).astype(bool),
        raw=raw,
        deseason=deseason,
        vol=vol,
    )


def strategy_weights(
    inputs: CarryInputs,
    strategy: str,
    signal_kind: str,
    start: str,
    end: str,
    *,
    config: CarryConfig,
) -> pd.DataFrame:
    """Target weights on the rebalance dates of ``[start, end)``."""
    days = inputs.returns.index[(inputs.returns.index >= start) & (inputs.returns.index < end)]
    rebal = rebalance_dates(pd.DatetimeIndex(days), config.rebalance_day)
    signal = (inputs.raw if signal_kind == "raw" else inputs.deseason).reindex(rebal)
    return target_weights(signal, inputs.vol.reindex(rebal), strategy, config)


def common_start(inputs: CarryInputs, not_before: str) -> str:
    """First date at which every asset has a seasonally adjusted signal and a volatility."""
    ready = inputs.deseason.notna().all(axis=1) & inputs.vol.notna().all(axis=1)
    ready = ready[ready.index >= not_before]
    first = pd.DatetimeIndex(ready.index[ready.to_numpy()])[0]
    return str(first.date())


def evaluate(
    inputs: CarryInputs,
    start: str,
    end: str,
    config: CarryConfig,
    costs_bps: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """Run every strategy × signal; return metrics, cost sensitivity and NAVs.

    Every strategy starts on the first rebalance day in ``[start, end)`` and is
    evaluated on the same window.
    """
    rows, sens = [], []
    navs: dict[str, pd.Series] = {}
    variants = [("TS", "raw"), ("TS", "deseason"), ("XS", "raw"), ("XS", "deseason"), ("LO", "raw")]
    for strategy, kind in variants:
        label = "LO" if strategy == "LO" else f"{strategy}-{kind}"
        w = strategy_weights(inputs, strategy, kind, start, end, config=config)
        first_trade = w.index.min()
        # the window starts on the first rebalance day (zero return, initial trade cost)
        window = inputs.returns.index[
            (inputs.returns.index >= first_trade) & (inputs.returns.index < end)
        ]
        rets = inputs.returns.reindex(window)
        for c in costs_bps:
            res = run_backtest(rets, inputs.rolled.reindex(window), w, c)
            perf = performance(res.daily["net"])
            if c == 0.0:
                years = perf["years"]
                rows.append(
                    {
                        "strategy": label,
                        **perf,
                        "turnover": float(res.daily["traded"].sum() / years),
                        "start": str(window.min().date()),
                        "end": str(window.max().date()),
                    }
                )
                navs[label] = (1.0 + res.daily["net"]).cumprod()
            sens.append({"strategy": label, "cost_bps": c, "sharpe": perf["sharpe"]})
    return pd.DataFrame(rows), pd.DataFrame(sens), navs


def robustness(
    slot2: pd.DataFrame,
    signals_by_variant: dict[str, pd.DataFrame],
    start: str,
    end: str,
    *,
    rebalance_days: tuple[int, ...] = (5, 15),
    vol_spans: tuple[int, ...] = (20, 120),
) -> pd.DataFrame:
    """Sharpe ratio (no costs) of the seasonally adjusted TS and XS strategies.

    Varies one choice at a time around the defaults: rebalance day, volatility
    span and the seasonal-adjustment variant. Shows how fragile a result is.
    """
    rows = []
    base = CarryConfig()
    grid: list[tuple[str, str, CarryConfig]] = [("default (day 10, span 60)", "shape", base)]
    grid += [(f"rebalance day {d}", "shape", CarryConfig(rebalance_day=d)) for d in rebalance_days]
    grid += [(f"vol span {v}", "shape", CarryConfig(vol_span=v)) for v in vol_spans]
    grid += [("full monthly mean removed", "anomaly", base)]
    for label, variant, cfg in grid:
        inputs = prepare_inputs(slot2, signals_by_variant[variant], cfg)
        for strategy in ("TS", "XS"):
            w = strategy_weights(inputs, strategy, "deseason", start, end, config=cfg)
            window = inputs.returns.index[
                (inputs.returns.index >= w.index.min()) & (inputs.returns.index < end)
            ]
            res = run_backtest(inputs.returns.reindex(window), inputs.rolled.reindex(window), w)
            perf = performance(res.daily["net"])
            rows.append(
                {
                    "variant": label,
                    "strategy": f"{strategy}-deseason",
                    "sharpe": perf["sharpe"],
                    "t_stat": perf["t_stat"],
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["variant", "strategy"])

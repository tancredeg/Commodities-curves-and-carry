"""Module 3a: storage contract, intrinsic value (LP and grid DP), forward curves."""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energycurves.storage.contract import (
    StorageContract,
    discount_factors,
    payment_dates,
    tbill_continuous_rate,
)
from energycurves.storage.curve import chained_seasonal_curve, seasonal_log_spreads, strip_forwards
from energycurves.storage.intrinsic import intrinsic_grid, intrinsic_lp, plan_value

BASE = StorageContract()
ONES = np.ones(BASE.n_months)


def _random_curves(n: int, months: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    level = rng.uniform(2.0, 5.0, size=(n, 1))
    season = 0.6 * np.cos(2 * np.pi * (np.arange(months) + rng.uniform(0, 12, (n, 1))) / 12)
    return level * np.exp(0.2 * season + rng.normal(0, 0.05, (n, months)))


def _brute_force(contract: StorageContract, fwd: np.ndarray, disc: np.ndarray) -> float:
    """Enumerate every grid schedule (small contracts only)."""
    step = contract.grid_step()
    assert step is not None
    acts = (
        np.arange(-round(contract.max_withdrawal / step), round(contract.max_injection / step) + 1)
        * step
    )
    best = -np.inf
    for combo in itertools.product(acts, repeat=contract.n_months):
        net = np.array(combo)
        inv = contract.initial + np.cumsum(net)
        if inv.min() < -1e-12 or inv.max() > contract.capacity + 1e-12:
            continue
        if abs(inv[-1] - contract.terminal) > 1e-12:
            continue
        best = max(best, float(plan_value(net, fwd, disc, contract)))
    return best


def test_grid_step_of_default_and_irregular_contracts() -> None:
    assert BASE.grid_step() == pytest.approx(1 / 6)
    assert replace(BASE, max_injection=0.1234567).grid_step() is None
    with pytest.raises(ValueError, match="inventories"):
        StorageContract(initial=2.0)


def test_flat_curve_with_costs_gives_zero_and_no_trade() -> None:
    fwd = 3.0 * ONES
    lp = intrinsic_lp(BASE, fwd, ONES)
    dp = intrinsic_grid(BASE, fwd, ONES)
    assert lp.value == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(lp.net, 0.0) and np.allclose(dp.net, 0.0)
    assert dp.value[0] == pytest.approx(0.0, abs=1e-12)


def test_two_month_closed_form() -> None:
    contract = StorageContract(start="2027-04", n_months=2, max_injection=0.5, max_withdrawal=1.0)
    disc = np.array([0.99, 0.98])
    for f1, f2 in [(2.0, 2.6), (3.0, 2.9), (2.5, 2.54)]:
        expected = 0.5 * max(disc[1] * (f2 - 0.02) - disc[0] * (f1 + 0.02), 0.0)
        assert intrinsic_lp(contract, np.array([f1, f2]), disc).value == pytest.approx(expected)
        assert intrinsic_grid(contract, np.array([f1, f2]), disc).value[0] == pytest.approx(
            expected
        )


def test_lp_equals_brute_force_on_three_months() -> None:
    contract = StorageContract(
        n_months=3, capacity=1.0, max_injection=0.5, max_withdrawal=0.5, cost_injection=0.01
    )
    rng = np.random.default_rng(3)
    for _ in range(20):
        fwd = rng.uniform(2.0, 4.0, 3)
        disc = np.exp(-0.04 * np.arange(1, 4) / 12)
        bf = _brute_force(contract, fwd, disc)
        assert intrinsic_lp(contract, fwd, disc).value == pytest.approx(bf, abs=1e-10)
        assert intrinsic_grid(contract, fwd, disc).value[0] == pytest.approx(bf, abs=1e-10)


def test_dp_equals_lp_on_random_year_long_curves() -> None:
    curves = _random_curves(40, BASE.n_months, seed=11)
    disc = np.exp(-0.04 * np.arange(1, 13) / 12)
    dp = intrinsic_grid(BASE, curves, disc)
    lp = np.array([intrinsic_lp(BASE, c, disc).value for c in curves])
    assert np.allclose(dp.value, lp, atol=1e-10)
    # the DP plan reproduces its own value
    assert np.allclose(plan_value(dp.net, curves, disc, BASE), dp.value, atol=1e-12)


def test_constraints_hold_on_optimal_plans() -> None:
    disc = np.exp(-0.04 * np.arange(1, 13) / 12)
    for curve in _random_curves(10, BASE.n_months, seed=5):
        plan = intrinsic_lp(BASE, curve, disc)
        assert plan.inventory.min() >= -1e-9
        assert plan.inventory.max() <= BASE.capacity + 1e-9
        assert plan.inventory[-1] == pytest.approx(BASE.terminal, abs=1e-9)
        assert plan.injection.max() <= BASE.max_injection + 1e-9
        assert plan.withdrawal.max() <= BASE.max_withdrawal + 1e-9
        assert np.all(plan.injection * plan.withdrawal < 1e-12)  # never both in one month


def test_value_increases_with_capacity_and_rates() -> None:
    curve = _random_curves(1, BASE.n_months, seed=9)[0]
    disc = ONES
    base = intrinsic_lp(BASE, curve, disc).value
    assert base > 0
    for field, factor in [("capacity", 1.5), ("max_injection", 1.5), ("max_withdrawal", 1.5)]:
        bigger = replace(BASE, **{field: getattr(BASE, field) * factor})
        assert intrinsic_lp(bigger, curve, disc).value >= base - 1e-12


def test_value_is_homogeneous_in_prices_and_costs() -> None:
    curve = _random_curves(1, BASE.n_months, seed=2)[0]
    disc = np.exp(-0.03 * np.arange(1, 13) / 12)
    base = intrinsic_lp(BASE, curve, disc).value
    for c in (0.5, 3.0):
        scaled = replace(
            BASE, cost_injection=BASE.cost_injection * c, cost_withdrawal=BASE.cost_withdrawal * c
        )
        assert intrinsic_lp(scaled, c * curve, disc).value == pytest.approx(c * base)


def test_dp_from_an_intermediate_month_equals_the_sub_problem() -> None:
    curve = _random_curves(1, BASE.n_months, seed=8)[0]
    disc = np.exp(-0.04 * np.arange(1, 13) / 12)
    for first, inv in [(3, 1 / 3), (6, 1.0), (9, 0.5)]:
        sub = replace(BASE, start=str(BASE.months[first]), n_months=12 - first, initial=inv)
        expected = intrinsic_lp(sub, curve[first:], disc[first:]).value
        got = intrinsic_grid(BASE, curve, disc, first_month=first, start_inventory=inv)
        assert got.value[0] == pytest.approx(expected, abs=1e-10)
        assert np.allclose(got.net[0, :first], 0.0)


def test_unreachable_terminal_and_off_grid_start_are_rejected() -> None:
    with pytest.raises(ValueError, match="unreachable"):
        intrinsic_grid(BASE, 3.0 * ONES, ONES, first_month=11, start_inventory=1.0)
    with pytest.raises(ValueError, match="grid"):
        intrinsic_grid(BASE, 3.0 * ONES, ONES, start_inventory=0.1)


def test_discounting_conventions() -> None:
    pay = payment_dates(BASE)
    assert pay[0] == pd.Timestamp("2027-04-15") and pay[-1] == pd.Timestamp("2028-03-15")
    d = discount_factors(BASE, "2026-11-04", 0.04)
    assert np.all(np.diff(d) < 0)
    days_last = (pd.Timestamp("2028-03-15") - pd.Timestamp("2026-11-04")).days
    assert d[-1] == pytest.approx(np.exp(-0.04 * days_last / 365))
    # 4 % discount yield on 91 days -> slightly above 4 % continuous Act/365
    r = tbill_continuous_rate(4.0)
    assert r == pytest.approx(-np.log(1 - 0.04 * 91 / 360) * 365 / 91)
    assert 0.040 < r < 0.041


def test_market_strip_and_seasonal_curve(sample_dir: Path, panel: pd.DataFrame) -> None:
    strip = pd.read_parquet(sample_dir / "ng_strip_snapshot.parquet")
    fwd = strip_forwards(strip, BASE)
    assert fwd.shape == (12,) and np.all(fwd > 0)
    with pytest.raises(ValueError, match="no price"):
        strip_forwards(strip, replace(BASE, start="2028-06"))
    spreads = seasonal_log_spreads(panel)
    assert list(spreads["month"]) == list(range(1, 13))
    assert spreads["n_obs"].min() > 100
    # winter premium: spreads into December/January are positive, out of January negative
    by_month = spreads.set_index("month")["mean_log_spread"]
    assert by_month[11] > 0 and by_month[1] < 0
    curve = chained_seasonal_curve(spreads, BASE, level=float(fwd.mean()))
    assert curve.mean() == pytest.approx(fwd.mean())
    assert np.allclose(np.diff(np.log(curve)), [by_month[m.month] for m in BASE.months[:-1]])

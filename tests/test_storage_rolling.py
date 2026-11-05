"""Module 3c: rolling-intrinsic valuation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energycurves.storage.contract import StorageContract, discount_factors
from energycurves.storage.forward_model import TwoFactorParams
from energycurves.storage.intrinsic import intrinsic_lp, plan_value
from energycurves.storage.rolling import (
    contract_expiries,
    decision_dates,
    rolling_intrinsic,
    sensitivity,
    value_storage,
)

CONTRACT = StorageContract()
T0 = "2026-11-04"
RATE = 0.04
PARAMS = TwoFactorParams(sigma_s=0.6, kappa=2.5, sigma_l=0.4, rho=-0.3)
CURVE = np.array([2.4, 2.45, 2.6, 2.85, 2.93, 2.92, 3.0, 3.3, 3.9, 4.35, 3.95, 3.25])


def test_decision_dates_cover_every_expiry() -> None:
    dates = decision_dates(CONTRACT, T0)
    expiries = contract_expiries(CONTRACT)
    assert dates[0] == pd.Timestamp(T0)
    assert dates.is_monotonic_increasing and dates.is_unique
    assert expiries.isin(dates).all()
    assert dates[-1] == expiries[-1]
    with pytest.raises(ValueError, match="precede"):
        decision_dates(CONTRACT, "2028-03-01")


def test_zero_volatility_gives_the_intrinsic_value() -> None:
    still = TwoFactorParams(sigma_s=0.0, kappa=1.0, sigma_l=0.0, rho=0.0)
    val, res = value_storage(
        CONTRACT, CURVE, T0, RATE, still, n_paths=8, rng=np.random.default_rng(0)
    )
    lp = intrinsic_lp(CONTRACT, CURVE, discount_factors(CONTRACT, T0, RATE)).value
    assert val.intrinsic == pytest.approx(lp, abs=1e-10)
    assert np.allclose(res.value, lp, atol=1e-10)
    assert val.extrinsic == pytest.approx(0.0, abs=1e-10)


def test_rolling_intrinsic_dominates_intrinsic_path_by_path() -> None:
    val, res = value_storage(
        CONTRACT, CURVE, T0, RATE, PARAMS, n_paths=2000, rng=np.random.default_rng(1)
    )
    assert res.gains[:, 1:].min() >= -1e-12
    assert np.all(res.value >= val.intrinsic - 1e-12)
    assert val.extrinsic > 3 * val.rolling_se  # re-optimising captures some value
    assert val.rolling_se > 0


def test_final_schedules_are_feasible() -> None:
    _, res = value_storage(
        CONTRACT, CURVE, T0, RATE, PARAMS, n_paths=500, rng=np.random.default_rng(2)
    )
    inv = CONTRACT.initial + np.cumsum(res.final_net, axis=1)
    assert inv.min() >= -1e-12 and inv.max() <= CONTRACT.capacity + 1e-12
    assert np.allclose(inv[:, -1], CONTRACT.terminal)
    assert res.final_net.max() <= CONTRACT.max_injection + 1e-12
    assert res.final_net.min() >= -CONTRACT.max_withdrawal - 1e-12


def test_telescoping_value_equals_physical_plus_hedge_cash_flows() -> None:
    """Independent replay with the LP solver and explicit hedge accounting.

    Executed schedule valued at the final prices of each month, plus the
    mark-to-market of the hedge held between consecutive decisions::

        Π(x_final; F_final) + Σ_j [Π(x_{j−1}; F_{j−1}) − Π(x_{j−1}; F_j)].
    """
    rng = np.random.default_rng(5)
    P, J, M = 40, 6, 4
    contract = StorageContract(n_months=M, max_injection=0.5, max_withdrawal=0.5)
    base = np.array([2.0, 2.4, 3.1, 2.8])
    curves = np.empty((P, J, M))
    curves[:, 0] = base
    locked_after = np.array([0, 0, 1, 2, 3, 4])
    for j in range(1, J):
        curves[:, j] = curves[:, j - 1] * np.exp(rng.normal(0, 0.15, (P, M)))
        frozen = locked_after[j - 1]  # a month stops moving once its volume is final
        curves[:, j, :frozen] = curves[:, j - 1, :frozen]
    disc = np.exp(-0.03 * np.arange(1, M + 1) / 12)
    res = rolling_intrinsic(contract, curves, disc, locked_after)

    for p in range(P):
        plan = np.zeros(M)
        hedge = 0.0
        previous = None
        for j in range(J):
            first = 0 if j == 0 else int(locked_after[j - 1])
            prices = curves[p, j]
            if previous is not None:
                hedge += float(
                    plan_value(previous, curves[p, j - 1], disc, contract)
                    - plan_value(previous, prices, disc, contract)
                )
            inventory = contract.initial + plan[:first].sum()
            sub = StorageContract(
                start=str(contract.months[first]),
                n_months=M - first,
                max_injection=0.5,
                max_withdrawal=0.5,
                initial=round(inventory * 2) / 2,
            )
            lp = intrinsic_lp(sub, prices[first:], disc[first:])
            plan = plan.copy()
            plan[first:] = lp.net
            previous = plan
        total = float(plan_value(plan, curves[p, -1], disc, contract)) + hedge
        assert total == pytest.approx(res.value[p], abs=1e-9)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="locked_after"):
        rolling_intrinsic(CONTRACT, np.ones((2, 3, 12)), np.ones(12), np.array([0, 5, 3]))


def test_sensitivity_uses_common_random_numbers() -> None:
    table = sensitivity(
        CONTRACT,
        CURVE,
        T0,
        RATE,
        {"base": PARAMS, "same again": PARAMS, "no short factor": PARAMS.scaled(sigma_s=0.0)},
        n_paths=400,
        seed=3,
    )
    base, again, flat = table["rolling_intrinsic"]
    assert base == again  # identical draws for identical parameters
    assert np.allclose(table["intrinsic"], table["intrinsic"].iloc[0])
    # without the short factor every contract month moves by the same factor:
    # the optimal schedule rarely changes, so re-optimising captures (almost) nothing
    assert 0.0 <= flat - table["intrinsic"].iloc[0] < base - table["intrinsic"].iloc[0]

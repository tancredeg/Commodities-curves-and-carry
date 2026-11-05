"""Rolling-intrinsic valuation of the storage contract (lower bound on its option value).

The intrinsic strategy locks today's best schedule and never changes it. The
rolling-intrinsic (RI) strategy re-optimises the schedule each time the curve
is observed again, keeping the months already delivered, and moves the hedge
to the new optimum. Decisions are taken on the valuation date and on every
NYMEX natural-gas expiry until the last contract month; on the expiry of a
contract month its volume is final.

With plans ``x_0, x_1, ...`` chosen at dates ``t_0 < t_1 < ...`` and
``Π(x; F)`` the discounted value of plan ``x`` on curve ``F``, the cash flow of
the hedged strategy telescopes (Gray & Khandelwal, 2004; Lai, Margot &
Secomandi, 2010)::

    RI = Π(x_0; F_{t_0}) + Σ_{j≥1} [ Π(x_j; F_{t_j}) − Π(x_{j−1}; F_{t_j}) ].

Each bracket is the gain of switching plan at today's prices, ≥ 0 because the
previous plan is still feasible: without transaction costs on the hedge, RI is
at least the intrinsic value on *every* path. The expectation of RI under the
model is a lower bound on the value of the storage option (it is not the
optimal exercise policy); ``RI − intrinsic`` is the extrinsic value captured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energycurves.data.calendar import last_trade_date
from energycurves.storage.contract import StorageContract, discount_factors
from energycurves.storage.forward_model import TwoFactorParams, simulate_curves
from energycurves.storage.intrinsic import intrinsic_grid, plan_value

DAYS_PER_YEAR = 365.0


def contract_expiries(contract: StorageContract, code: str = "NG") -> pd.DatetimeIndex:
    """Last trading day of the futures contract of each contract month."""
    return pd.DatetimeIndex([last_trade_date(code, m) for m in contract.months])


def decision_dates(
    contract: StorageContract, valuation_date: str | pd.Timestamp, code: str = "NG"
) -> pd.DatetimeIndex:
    """Valuation date, then every NYMEX expiry after it up to the last contract month."""
    t0 = pd.Timestamp(valuation_date)
    first = pd.Period(t0, freq="M")
    months = pd.period_range(first, contract.months[-1], freq="M")
    expiries = [last_trade_date(code, m) for m in months]
    later = sorted(d for d in expiries if d > t0)
    if not later or later[-1] != contract_expiries(contract, code)[-1]:
        raise ValueError("valuation date must precede the last contract expiry")
    return pd.DatetimeIndex([t0, *later])


@dataclass(frozen=True)
class RollingResult:
    """Per-path RI values and the final schedules."""

    value: np.ndarray  # (P,) discounted RI value per path
    intrinsic: float  # value of the plan locked at t_0
    final_net: np.ndarray  # (P, M) executed net injections
    gains: np.ndarray  # (P, J) re-optimisation gains, column 0 = intrinsic


def rolling_intrinsic(
    contract: StorageContract,
    curves: np.ndarray,
    discount: np.ndarray,
    locked_after: np.ndarray,
) -> RollingResult:
    """Run the RI strategy on simulated curves.

    ``curves``: ``(P, J, M)`` futures prices of the contract months at each
    decision date (``curves[:, 0]`` = today's curve, identical on every path).
    ``locked_after[j]``: number of leading contract months whose volume is
    final once the decision of date ``j`` is taken (their contract has expired
    on or before that date).
    """
    P, J, M = curves.shape
    if len(locked_after) != J or np.any(np.diff(locked_after) < 0) or locked_after[-1] != M:
        raise ValueError("locked_after must be non-decreasing and end at the number of months")
    net = np.zeros((P, M))
    gains = np.zeros((P, J))
    locked = 0
    for j in range(J):
        prices = curves[:, j, :]
        inventory = contract.initial + net[:, :locked].sum(axis=1)
        sol = intrinsic_grid(
            contract, prices, discount, first_month=locked, start_inventory=inventory
        )
        new_net = net.copy()
        new_net[:, locked:] = sol.net[:, locked:]
        if j == 0:
            gains[:, 0] = plan_value(new_net, prices, discount, contract)
        else:
            gains[:, j] = plan_value(new_net, prices, discount, contract) - plan_value(
                net, prices, discount, contract
            )
        net = new_net
        locked = int(locked_after[j])
    return RollingResult(
        value=gains.sum(axis=1), intrinsic=float(gains[0, 0]), final_net=net, gains=gains
    )


@dataclass(frozen=True)
class StorageValuation:
    """Intrinsic and rolling-intrinsic values (USD/MMBtu of capacity) with MC error."""

    intrinsic: float
    rolling_mean: float
    rolling_se: float
    n_paths: int
    n_decisions: int

    @property
    def extrinsic(self) -> float:
        return self.rolling_mean - self.intrinsic

    @property
    def extrinsic_share(self) -> float:
        """Extrinsic value as a fraction of the intrinsic value."""
        return self.extrinsic / self.intrinsic if self.intrinsic > 0 else float("nan")


def value_storage(
    contract: StorageContract,
    forwards: np.ndarray,
    valuation_date: str | pd.Timestamp,
    rate: float,
    params: TwoFactorParams,
    *,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[StorageValuation, RollingResult]:
    """Simulate the curve between decision dates and value the contract by RI.

    The Monte Carlo standard error uses the means of antithetic pairs, which
    are independent of each other.
    """
    t0 = pd.Timestamp(valuation_date)
    dates = decision_dates(contract, t0)
    expiries = contract_expiries(contract)
    times = (dates[1:] - t0).days.to_numpy() / DAYS_PER_YEAR
    expiry_times = (expiries - t0).days.to_numpy() / DAYS_PER_YEAR
    sims = simulate_curves(params, forwards, expiry_times, times, n_paths=n_paths, rng=rng)
    today = np.broadcast_to(np.asarray(forwards, dtype=float), (n_paths, 1, contract.n_months))
    curves = np.concatenate([today, sims], axis=1)
    locked_after = np.array([(expiries <= d).sum() for d in dates])
    discount = discount_factors(contract, t0, rate)
    result = rolling_intrinsic(contract, curves, discount, locked_after)
    half = n_paths // 2
    pairs = 0.5 * (result.value[:half] + result.value[half:])
    valuation = StorageValuation(
        intrinsic=result.intrinsic,
        rolling_mean=float(result.value.mean()),
        rolling_se=float(pairs.std(ddof=1) / np.sqrt(half)),
        n_paths=n_paths,
        n_decisions=len(dates),
    )
    return valuation, result


def sensitivity(
    contract: StorageContract,
    forwards: np.ndarray,
    valuation_date: str | pd.Timestamp,
    rate: float,
    variants: dict[str, TwoFactorParams],
    *,
    n_paths: int,
    seed: int,
) -> pd.DataFrame:
    """RI value under several parameter sets, with common random numbers (same seed)."""
    rows = []
    for label, params in variants.items():
        val, _ = value_storage(
            contract,
            forwards,
            valuation_date,
            rate,
            params,
            n_paths=n_paths,
            rng=np.random.default_rng(seed),
        )
        rows.append(
            {
                "variant": label,
                "sigma_s": params.sigma_s,
                "kappa": params.kappa,
                "sigma_l": params.sigma_l,
                "rho": params.rho,
                "intrinsic": val.intrinsic,
                "rolling_intrinsic": val.rolling_mean,
                "se": val.rolling_se,
                "extrinsic": val.extrinsic,
                "extrinsic_share": val.extrinsic_share,
            }
        )
    return pd.DataFrame(rows)

"""Natural-gas storage contract: capacity, rates, costs and the cash-flow calendar.

A storage contract gives the right to inject gas (buy) and withdraw gas (sell)
over a sequence of delivery months, subject to::

    V_m = V_{m-1} + i_m − w_m,   0 ≤ V_m ≤ V̄,   0 ≤ i_m ≤ Ī,   0 ≤ w_m ≤ W̄,
    V_0 = initial,   V_M = terminal.

Volumes are expressed per unit of working-gas capacity (V̄ = 1 by default), so
a value in USD/MMBtu is a value per MMBtu of capacity. The decision for month
``m`` is a monthly volume traded at the price of the futures contract for that
delivery month: the model has monthly granularity, with no intra-month spot
optionality and no inventory-dependent rates (ratchets).

Defaults follow a stylised seasonal (salt-cavern-free) reservoir: six months to
fill, three months to empty, 0.02 USD/MMBtu variable cost on each leg. They
are illustrative and every field is a parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StorageContract:
    """Storage contract over ``n_months`` consecutive delivery months from ``start``."""

    start: str = "2027-04"
    n_months: int = 12
    capacity: float = 1.0
    max_injection: float = 1.0 / 6.0
    max_withdrawal: float = 1.0 / 3.0
    initial: float = 0.0
    terminal: float = 0.0
    cost_injection: float = 0.02
    cost_withdrawal: float = 0.02

    def __post_init__(self) -> None:
        if self.n_months < 1:
            raise ValueError("n_months must be positive")
        if min(self.capacity, self.max_injection, self.max_withdrawal) <= 0:
            raise ValueError("capacity and rates must be positive")
        if not 0 <= self.initial <= self.capacity or not 0 <= self.terminal <= self.capacity:
            raise ValueError("initial and terminal inventories must lie in [0, capacity]")
        if min(self.cost_injection, self.cost_withdrawal) < 0:
            raise ValueError("costs must be non-negative")

    @property
    def months(self) -> pd.PeriodIndex:
        """Delivery months covered by the contract."""
        return pd.period_range(self.start, periods=self.n_months, freq="M")

    def grid_step(self, max_denominator: int = 120) -> float | None:
        """Largest step δ such that capacity, rates and boundary inventories are multiples of δ.

        Returns ``None`` when no such step exists with at most ``max_denominator``
        grid points per unit (the dynamic-programming solver then cannot be exact).
        """
        values = [
            self.capacity,
            self.max_injection,
            self.max_withdrawal,
            self.initial,
            self.terminal,
        ]
        fracs = [Fraction(v).limit_denominator(max_denominator) for v in values]
        if any(abs(float(f) - v) > 1e-12 for f, v in zip(fracs, values, strict=True)):
            return None
        denominator = math.lcm(*(f.denominator for f in fracs))
        numerators = [int(f * denominator) for f in fracs]
        step = Fraction(math.gcd(*numerators), denominator)
        return float(step)


def payment_dates(contract: StorageContract) -> pd.DatetimeIndex:
    """Cash-flow date of each delivery month: the 15th (mid-month delivery convention)."""
    starts = [m.to_timestamp(how="start") for m in contract.months]
    return pd.DatetimeIndex([s + pd.Timedelta(days=14) for s in starts])


def discount_factors(
    contract: StorageContract, valuation_date: str | pd.Timestamp, rate: float
) -> np.ndarray:
    """``D_m = exp(−r (t_m − t_0))`` with ``r`` continuously compounded, Act/365."""
    t0 = pd.Timestamp(valuation_date)
    years = (payment_dates(contract) - t0).days.to_numpy() / 365.0
    return np.exp(-rate * years)


def tbill_continuous_rate(discount_rate_pct: float, days: int = 91) -> float:
    """Continuously compounded Act/365 rate of a T-bill quoted on a discount basis.

    FRED ``DTB3`` is the discount yield ``d`` (Act/360): the price is
    ``P = 1 − d·days/360`` and ``r = −ln P / (days/365)``.
    """
    price = 1.0 - discount_rate_pct / 100.0 * days / 360.0
    return -math.log(price) / (days / 365.0)

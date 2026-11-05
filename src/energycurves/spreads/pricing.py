"""Crack-spread option grid: Kirk (two legs and basket) against Monte Carlo.

Spreads in USD/bbl (1 bbl = 42 gal), long products, short crude:

* heating-oil crack ``42·HO − CL`` (two legs: Kirk's own error),
* 3-2-1 crack ``(2·42·RB + 42·HO)/3 − CL`` (basket proxy + Kirk).

For each delivery month, the option expires on the crude oil last trading day
of that month (the earliest leg expiry); gasoline and heating oil expire at
the end of the previous month, later than crude. Strikes are set around the
forward spread. Both methods use the same calibrated one-factor legs and
correlations, so ``Kirk − MC`` measures the approximation error only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energycurves.data.registry import GALLONS_PER_BARREL
from energycurves.spreads.kirk import crack_call_kirk, kirk_call
from energycurves.spreads.montecarlo import spread_option_mc, terminal_draws
from energycurves.spreads.schwartz import LegParams, terminal_log_cov

LEGS = ("CL", "RB", "HO")
G = GALLONS_PER_BARREL
SPREADS: dict[str, np.ndarray] = {
    "HO crack (42 HO - CL)": np.array([-1.0, 0.0, G]),
    "3-2-1 crack": np.array([-1.0, 2.0 * G / 3.0, G / 3.0]),
}


@dataclass(frozen=True)
class MarketSnapshot:
    """Futures prices and expiries of the three legs for one delivery month."""

    delivery_month: pd.Timestamp
    forwards: np.ndarray  # CL, RB, HO in their own units
    futures_expiry: np.ndarray  # years from valuation date
    option_expiry: float  # years from valuation date


def snapshot(panel: pd.DataFrame, valuation_date: str, delivery_month: str) -> MarketSnapshot:
    """Prices of CL, RB, HO for ``delivery_month`` on ``valuation_date`` (from the curve panel)."""
    t0 = pd.Timestamp(valuation_date)
    month = pd.Timestamp(delivery_month)
    rows = panel[(panel["date"] == t0) & (panel["delivery_month"] == month)].set_index("code")
    missing = [c for c in LEGS if c not in rows.index]
    if missing:
        raise ValueError(f"no {delivery_month} price on {valuation_date} for {missing}")
    expiry = (rows.loc[list(LEGS), "last_trade_date"] - t0).dt.days.to_numpy() / 365.0
    return MarketSnapshot(
        delivery_month=month,
        forwards=rows.loc[list(LEGS), "settle"].to_numpy(dtype=float),
        futures_expiry=expiry,
        option_expiry=float(expiry.min()),
    )


def price_grid(
    snapshots: list[MarketSnapshot],
    legs: list[LegParams],
    corr: np.ndarray,
    rate: float,
    strike_offsets: tuple[float, ...],
    *,
    n_paths: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Kirk and Monte Carlo prices for every spread, delivery month and strike."""
    rows = []
    for snap in snapshots:
        cov = terminal_log_cov(legs, corr, snap.futures_expiry, snap.option_expiry)
        disc = float(np.exp(-rate * snap.option_expiry))
        draws = terminal_draws(snap.forwards, cov, n_paths, rng)
        for name, w in SPREADS.items():
            forward_spread = float(w @ snap.forwards)
            for offset in strike_offsets:
                K = forward_spread + offset
                if name.startswith("HO"):
                    kirk = kirk_call(
                        G * snap.forwards[2],
                        snap.forwards[0],
                        K,
                        v1=float(np.sqrt(cov[2, 2])),
                        v2=float(np.sqrt(cov[0, 0])),
                        c12=float(cov[0, 2]),
                        D=disc,
                    )
                else:
                    long_w = np.where(w > 0, w, 0.0)
                    kirk = crack_call_kirk(long_w, snap.forwards, cov, short=0, K=K, D=disc)
                mc = spread_option_mc(w, draws, K, disc)
                rows.append(
                    {
                        "spread": name,
                        "delivery_month": snap.delivery_month.strftime("%Y-%m"),
                        "option_expiry_years": snap.option_expiry,
                        "forward_spread": forward_spread,
                        "strike": K,
                        "moneyness": offset,
                        "mc_price": mc.price,
                        "mc_se": mc.se,
                        "kirk_price": kirk,
                        "kirk_minus_mc": kirk - mc.price,
                        "error_in_se": (kirk - mc.price) / mc.se,
                        "error_pct": 100.0 * (kirk - mc.price) / mc.price,
                    }
                )
    return pd.DataFrame(rows)

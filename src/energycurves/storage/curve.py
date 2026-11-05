"""Forward curves used to value the storage contract.

Two curves over the contract months, both in USD/MMBtu:

(a) **Market strip** — the snapshot of every live NYMEX Henry Hub contract
    (``data/sample/ng_strip_snapshot.parquet``, Yahoo closes). This is the
    curve a desk would value the contract on.

(b) **Average seasonal shape** — rebuilt from the EIA contracts 1–4 only
    (2024 and earlier), by chaining the mean adjacent calendar spreads::

        s_c = mean_t [ ln F_t(m+1) − ln F_t(m) ],   c = calendar month of m,
        ln F(m_k) = ln F(m_0) + Σ_{j<k} s_{c(m_j)},

    then scaling the curve so that its arithmetic mean equals a chosen level
    (by default the mean of the market strip over the same months). The EIA
    series only reach four months ahead, so a year-long curve can only come
    from chaining short spreads. It isolates the *typical* seasonal shape from
    the level and from today's specific spreads: it is a robustness check, not
    a market curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energycurves.storage.contract import StorageContract

SPREAD_WINDOW = ("2014-04-06", "2024-04-06")


def strip_forwards(strip: pd.DataFrame, contract: StorageContract) -> np.ndarray:
    """Market forwards of the contract months, from a snapshot ``delivery_month, settle``."""
    by_month = strip.set_index(pd.PeriodIndex(strip["delivery_month"], freq="M"))["settle"]
    missing = [m for m in contract.months if m not in by_month.index]
    if len(missing) > 0:
        raise ValueError(f"strip has no price for {', '.join(map(str, missing))}")
    out: np.ndarray = by_month.reindex(contract.months).to_numpy(dtype=float)
    return out


def seasonal_log_spreads(
    panel: pd.DataFrame,
    code: str = "NG",
    start: str = SPREAD_WINDOW[0],
    end: str = SPREAD_WINDOW[1],
) -> pd.DataFrame:
    """Mean of ``ln F(m+1) − ln F(m)`` over adjacent EIA contracts, by calendar month of ``m``.

    Uses every pair of adjacent slots (1-2, 2-3, 3-4) on every day of
    ``[start, end)`` whose delivery months are consecutive and prices positive.
    Returns ``month, n_obs, mean_log_spread, sd_log_spread``.
    """
    sub = panel[(panel["code"] == code) & (panel["date"] >= start) & (panel["date"] < end)]
    wide = sub.pivot(index="date", columns="contract", values=["settle", "delivery_month"])
    pieces = []
    for k in (1, 2, 3):
        near = wide[("settle", k)].astype(float)
        far = wide[("settle", k + 1)].astype(float)
        m_near = pd.PeriodIndex(wide[("delivery_month", k)], freq="M")
        m_far = pd.PeriodIndex(wide[("delivery_month", k + 1)], freq="M")
        consecutive = np.asarray(m_far == m_near + 1)
        ok = consecutive & (near > 0).to_numpy() & (far > 0).to_numpy()
        pieces.append(
            pd.DataFrame(
                {
                    "month": m_near.month[ok],
                    "log_spread": np.log(far.to_numpy()[ok] / near.to_numpy()[ok]),
                }
            )
        )
    allp = pd.concat(pieces, ignore_index=True)
    out = (
        allp.groupby("month")["log_spread"]
        .agg(n_obs="size", mean_log_spread="mean", sd_log_spread="std")
        .reindex(range(1, 13))
    )
    if out["n_obs"].isna().any():
        raise ValueError("some calendar months have no adjacent spread in the window")
    return out.reset_index()


def chained_seasonal_curve(
    spreads: pd.DataFrame, contract: StorageContract, level: float
) -> np.ndarray:
    """Chain the mean monthly log spreads over the contract months; arithmetic mean = ``level``."""
    by_month = spreads.set_index("month")["mean_log_spread"]
    months = contract.months
    log_curve = np.zeros(len(months))
    for k in range(1, len(months)):
        log_curve[k] = log_curve[k - 1] + by_month[months[k - 1].month]
    shape = np.exp(log_curve)
    out: np.ndarray = level * shape / shape.mean()
    return out

"""Constant-contract returns and the clean curve panel.

A naive ``pct_change`` on the "contract 1" series mixes two different
contracts on every roll day, so the return contains the calendar spread (the
roll yield itself). Here every return is computed on *one* contract::

    r_t = F_t(D) / F_{t-1}(D) − 1,   D = contract held at the close of t−1.

For a position kept in slot ``k`` (e.g. always the second contract), ``D`` is
the delivery month in slot ``k`` on day ``t−1``; if contract 1 expired on
``t−1``, ``D`` sits in slot ``k−1`` on ``t`` and the position is re-established
in slot ``k`` at the close of ``t``. This is the excess return of a fully
collateralised rolling futures position (Gorton & Rouwenhorst, 2006).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energycurves.data.calendar import assign_delivery_months, business_days, expiry_calendar
from energycurves.data.clean import drop_non_business_days, flag_isolated_spikes
from energycurves.data.registry import CODES


def build_curve_panel(futures: pd.DataFrame) -> pd.DataFrame:
    """Raw EIA slot prices -> clean panel with delivery months and time to expiry.

    Steps: drop rows on non-business days, attach ``delivery_month``,
    ``last_trade_date``, ``tau`` (Act/365) from the expiry calendar, drop
    isolated spikes on contracts 2-4. Output columns: ``date, code, contract,
    delivery_month, last_trade_date, tau, settle``.
    """
    clean, _ = drop_non_business_days(futures, business_days())
    calendar = expiry_calendar(CODES, "1979-01", "2031-12")
    panel = assign_delivery_months(clean[["date", "code", "contract", "settle"]], calendar)
    panel = panel[~flag_isolated_spikes(panel)]
    cols = ["date", "code", "contract", "delivery_month", "last_trade_date", "tau", "settle"]
    return panel[cols].sort_values(["code", "date", "contract"], ignore_index=True)


PRICE_LOOKBACK = 5  # trading days: oldest settle of the held contract used as p0


def slot_returns(panel: pd.DataFrame, slot: int) -> pd.DataFrame:
    """Daily constant-contract return of a position rolled to stay in ``slot``.

    Returns ``date, code, ret, held_month, rolled``. The slot occupant comes
    from the curve structure, not from the presence of a price: on each day,
    contract ``c`` in month ``m`` puts the front month at ``m − (c − 1)``, so
    the month in ``slot`` is known even when its settle is missing.
    ``held_month`` is the month in the slot on ``t−1``; ``rolled`` is True when
    another month occupies the slot on ``t`` (an expiry in between).

    ``ret = F_t(D) / F_s(D) − 1`` with ``s`` the last session ``≤ t−1`` where
    ``D`` has a settle, at most ``PRICE_LOOKBACK − 1`` sessions before ``t−1``
    and not before ``D`` entered the slot: a missing settle delays the return,
    it does not lose it. ``ret`` is NaN when ``D`` has no settle on ``t``, no
    usable ``F_s`` or ``F_s ≤ 0``.
    """
    out = []
    for code, grp in panel.groupby("code", sort=True):
        days = pd.DatetimeIndex(np.sort(grp["date"].unique()))
        day_no = pd.Series(np.arange(len(days)), index=days).reindex(grp["date"]).to_numpy()
        month = grp["delivery_month"].to_numpy().astype("datetime64[M]").astype(np.int64)
        front_of_row = pd.Series(month - (grp["contract"].to_numpy() - 1))
        front = front_of_row.groupby(day_no).agg(["min", "max"])
        if (front["min"] != front["max"]).any():
            raise ValueError(f"{code}: slots disagree on the front month")
        occupant = front["min"].to_numpy() + slot - 1  # month in the slot, every day
        key = pd.MultiIndex.from_arrays([day_no, month])
        price = pd.Series(grp["settle"].to_numpy(), index=key)
        j = np.arange(len(days) - 1)  # index of t−1
        held = occupant[:-1]
        p1 = price.reindex(pd.MultiIndex.from_arrays([j + 1, held])).to_numpy()
        start = np.maximum(np.searchsorted(occupant, held, side="left"), j - (PRICE_LOOKBACK - 1))
        p0 = np.full(len(j), np.nan)
        for lag in range(PRICE_LOOKBACK):  # most recent usable settle of the held contract
            cand = price.reindex(pd.MultiIndex.from_arrays([j - lag, held])).to_numpy()
            p0 = np.where(np.isnan(p0) & (j - lag >= start) & ~np.isnan(cand), cand, p0)
        valid = (np.nan_to_num(p0, nan=0.0) > 0) & np.isfinite(p1)
        frame = pd.DataFrame({"date": days[1:], "code": code})
        frame["ret"] = np.where(valid, p1 / np.where(valid, p0, 1.0) - 1.0, np.nan)
        frame["held_month"] = held.astype("datetime64[M]").astype("datetime64[ns]")
        frame["rolled"] = occupant[1:] != held
        out.append(frame[["date", "code", "ret", "held_month", "rolled"]])
    return pd.concat(out, ignore_index=True)


def excess_return_index(returns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative index ``Π(1 + r)`` per code (missing returns count as 0)."""
    wide = returns.pivot(index="date", columns="code", values="ret").fillna(0.0)
    return (1.0 + wide).cumprod()

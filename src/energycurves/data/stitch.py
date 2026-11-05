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


def slot_returns(panel: pd.DataFrame, slot: int) -> pd.DataFrame:
    """Daily constant-contract return of a position rolled to stay in ``slot``.

    Returns ``date, code, ret, held_month, rolled`` where ``held_month`` is the
    delivery month held over ``(t−1, t]`` and ``rolled`` is True when that
    contract left the slot on ``t`` (the position is rolled at the close of
    ``t``). ``ret`` is NaN when the held contract has no price on ``t`` or a
    non-positive price on ``t−1``.
    """
    out = []
    for code, grp in panel.groupby("code", sort=True):
        days = pd.DatetimeIndex(np.sort(grp["date"].unique()))
        prev_day = pd.Series(days[:-1], index=days[1:])
        in_slot = grp[grp["contract"] == slot].set_index("date")["delivery_month"]
        price = grp.set_index(["date", "delivery_month"])["settle"]
        slot_of = grp.set_index(["date", "delivery_month"])["contract"]
        frame = pd.DataFrame({"date": days[1:], "prev": prev_day.to_numpy()})
        frame["held_month"] = in_slot.reindex(frame["prev"]).to_numpy()
        key_prev = pd.MultiIndex.from_arrays([frame["prev"], frame["held_month"]])
        key_now = pd.MultiIndex.from_arrays([frame["date"], frame["held_month"]])
        p0 = price.reindex(key_prev).to_numpy()
        p1 = price.reindex(key_now).to_numpy()
        valid = (p0 > 0) & np.isfinite(p1)
        frame["ret"] = np.where(valid, p1 / np.where(p0 > 0, p0, np.nan) - 1.0, np.nan)
        frame["rolled"] = slot_of.reindex(key_now).to_numpy() != slot
        frame["code"] = code
        out.append(frame[["date", "code", "ret", "held_month", "rolled"]])
    return pd.concat(out, ignore_index=True)


def excess_return_index(returns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative index ``Π(1 + r)`` per code (missing returns count as 0)."""
    wide = returns.pivot(index="date", columns="code", values="ret").fillna(0.0)
    return (1.0 + wide).cumprod()

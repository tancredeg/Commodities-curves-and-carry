"""NYMEX business-day calendar, last trading days and delivery-month assignment.

The EIA publishes prices by *contract slot* (contract 1 = nearest delivery
month still trading, contracts 2-4 = the following months). To compute roll
yields and same-contract returns we need, for every trade date, the delivery
month and the time to expiry of each slot. This module rebuilds them from the
exchange rules quoted in the EIA definitions:

* CL: trading ceases 3 business days before the 25th calendar day of the
  month preceding delivery; if the 25th is not a business day, 3 business days
  before the last business day preceding the 25th.
* NG: trading ceases 3 business days before the first calendar day of the
  delivery month (CME rulebook, chapter 220); earlier in the 1990s, see
  ``_ng_business_days_before``.
* RB, HO: trading ceases on the last business day of the month preceding
  delivery.

Contract 1 on date ``t`` is the first delivery month whose last trading day is
``>= t``: on its last trading day the expiring contract is still contract 1.

Business days follow the NYSE/NYMEX holiday schedule (US federal holidays
without Columbus and Veterans Day, plus Good Friday) and a short list of
unscheduled closures. ``tests/test_calendar.py`` checks the calendar against the
trading days and roll dates observed in the EIA data.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import numpy as np
import pandas as pd
from dateutil.relativedelta import MO
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import DateOffset

DAYS_PER_YEAR = 365.0

# Unscheduled full-day closures of the New York exchanges.
SPECIAL_CLOSURES: tuple[str, ...] = (
    "1994-04-27",  # National day of mourning, President Nixon
    "2001-09-11",  # September 11 attacks (NYMEX reopened on 2001-09-14)
    "2001-09-12",
    "2001-09-13",
    "2004-06-11",  # National day of mourning, President Reagan
    "2007-01-02",  # National day of mourning, President Ford
    "2025-01-09",  # National day of mourning, President Carter
)
# Days closed for the NYSE but on which the EIA reports genuine NYMEX
# settlements (electronic trading): 1985-09-27, 2012-10-29/30 (Hurricane Sandy),
# 2018-12-05. They are treated as business days.


# Exchange days that the 2008-2010 NG expiries did not count as business days
# (day after Thanksgiving, New Year's Eve 2010): the EIA contract slots shift one
# trading day earlier than the rule would imply in those four months.
NG_EXPIRY_NON_BUSINESS_DAYS: tuple[str, ...] = (
    "2008-11-28",
    "2009-11-27",
    "2010-11-26",
    "2010-12-31",
)


class NymexHolidayCalendar(AbstractHolidayCalendar):
    """Scheduled NYMEX (NYSE-like) full-day holidays."""

    rules = [  # noqa: RUF012 - pandas API expects a class attribute list
        Holiday("NewYearsDay", month=1, day=1, observance=sunday_to_monday),
        # The exchanges observe Martin Luther King Jr. Day since 1998 only.
        Holiday(
            "MLKDay", month=1, day=1, start_date="1998-01-01", offset=DateOffset(weekday=MO(3))
        ),
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


@lru_cache(maxsize=8)
def _holidays(start_year: int, end_year: int) -> frozenset[pd.Timestamp]:
    cal = NymexHolidayCalendar()
    scheduled = cal.holidays(pd.Timestamp(start_year, 1, 1), pd.Timestamp(end_year, 12, 31))
    special = pd.DatetimeIndex(pd.to_datetime(list(SPECIAL_CLOSURES)))
    return frozenset(scheduled.append(special))


@lru_cache(maxsize=8)
def business_days(start: str = "1980-01-01", end: str = "2035-12-31") -> pd.DatetimeIndex:
    """Weekdays minus NYMEX holidays between ``start`` and ``end`` (inclusive)."""
    days = pd.bdate_range(start, end)
    hol = _holidays(days[0].year, days[-1].year)
    return days[~days.isin(list(hol))]


def _is_business_day(day: pd.Timestamp) -> bool:
    return bool(business_days().isin([day]).any())


def _business_days_before(day: pd.Timestamp, n: int) -> pd.Timestamp:
    """The ``n``-th business day strictly before ``day``."""
    bd = business_days()
    pos = int(bd.searchsorted(day, side="left"))
    return pd.Timestamp(bd[pos - n])


def _ng_business_days_before(delivery_month: pd.Period) -> int:
    """Business days between NG expiry and the first day of the delivery month.

    Current NYMEX rule: 3. The EIA prices show earlier expiries in the 1990s
    (inferred from the day the contract slots shift, see ``detect_rolls``):
    6 business days for deliveries up to January 1996, 5 business days for
    February 1996 to January 1997.
    """
    if delivery_month <= pd.Period("1996-01", freq="M"):
        return 6
    if delivery_month <= pd.Period("1997-01", freq="M"):
        return 5
    return 3


def last_trade_date(code: str, delivery_month: pd.Period) -> pd.Timestamp:
    """Last trading day of a NYMEX energy futures contract (rules in the module docstring)."""
    prev = (delivery_month - 1).to_timestamp(how="start")
    month_start = delivery_month.to_timestamp(how="start")
    bd = business_days()
    if code == "CL":
        the_25th = prev + pd.Timedelta(days=24)
        anchor = the_25th
        if not _is_business_day(the_25th):
            # last business day preceding the 25th
            anchor = pd.Timestamp(bd[int(bd.searchsorted(the_25th, side="left")) - 1])
        return _business_days_before(anchor, 3)
    if code == "NG":
        n = _ng_business_days_before(delivery_month)
        bd_ng = bd[~bd.isin(pd.to_datetime(list(NG_EXPIRY_NON_BUSINESS_DAYS)))]
        pos = int(bd_ng.searchsorted(month_start, side="left"))
        return pd.Timestamp(bd_ng[pos - n])
    if code in ("RB", "HO"):
        return _business_days_before(month_start, 1)
    raise ValueError(f"unknown commodity code {code!r}")


def expiry_calendar(
    codes: Iterable[str], first_month: str = "1983-01", last_month: str = "2030-12"
) -> pd.DataFrame:
    """Table ``(code, delivery_month, last_trade_date)`` for every listed month.

    ``delivery_month`` is the first calendar day of the delivery month.
    """
    months = pd.period_range(first_month, last_month, freq="M")
    rows = [
        (code, p.to_timestamp(how="start"), last_trade_date(code, p))
        for code in codes
        for p in months
    ]
    out = pd.DataFrame(rows, columns=["code", "delivery_month", "last_trade_date"])
    out["delivery_month"] = out["delivery_month"].astype("datetime64[ns]")
    out["last_trade_date"] = out["last_trade_date"].astype("datetime64[ns]")
    return out


def assign_delivery_months(prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Attach ``delivery_month``, ``last_trade_date`` and ``tau`` to a slot panel.

    ``prices`` has columns ``date, code, contract`` (contract = 1..4). Contract
    ``k`` on ``date`` is the ``k``-th delivery month whose last trading day is
    on or after ``date``. ``tau = (last_trade_date - date) / 365`` (Act/365).
    """
    pieces = []
    for code, grp in prices.groupby("code", sort=True):
        cal = calendar[calendar["code"] == code].sort_values("delivery_month")
        ltd = cal["last_trade_date"].to_numpy(dtype="datetime64[ns]")
        first_live = np.searchsorted(ltd, grp["date"].to_numpy(dtype="datetime64[ns]"), side="left")
        idx = first_live + grp["contract"].to_numpy() - 1
        if (idx >= len(cal)).any():
            raise ValueError(f"calendar too short for {code}")
        out = grp.copy()
        out["delivery_month"] = cal["delivery_month"].to_numpy()[idx]
        out["last_trade_date"] = ltd[idx]
        pieces.append(out)
    res = pd.concat(pieces, ignore_index=True)
    res["tau"] = (res["last_trade_date"] - res["date"]).dt.days / DAYS_PER_YEAR
    return res


def roll_dates(
    calendar: pd.DataFrame, code: str, trading_days: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """First trading day after each last trading day: the day contract slots shift."""
    ltd = calendar.loc[calendar["code"] == code, "last_trade_date"]
    pos = trading_days.searchsorted(pd.DatetimeIndex(ltd), side="right")
    pos = pos[pos < len(trading_days)]
    return pd.DatetimeIndex(np.unique(trading_days[pos]))


def price_implied_roll_offsets(
    wide: pd.DataFrame, expected_rolls: pd.DatetimeIndex, window: int = 4
) -> pd.DataFrame:
    """Locate each slot shift from prices alone and compare it with the calendar.

    ``wide``: one commodity, indexed by trading day, columns 1..4 = contracts 1-4.
    With ``S^k_t = ln F^{k+1}_t − ln F^k_t`` (adjacent calendar spreads), on the
    day the slots shift today's spreads 1-2 equal yesterday's spreads 2-3::

        e_same(t)  = |S^2_t − S^2_{t−1}| + |S^3_t − S^3_{t−1}|
        e_shift(t) = |S^1_t − S^2_{t−1}| + |S^2_t − S^3_{t−1}|

    The expiring contract never enters the statistic (its price converges on
    its last day and would blur the test). Assuming exactly one shift within
    ``±window`` trading days of the expected roll, the best day minimises
    ``e_shift − e_same``. ``snr`` divides the gap between the best and second
    best day by the median daily spread change of the past year: rolls of
    nearly flat curves (``snr`` small) are not identifiable from prices.

    Returns ``(expected_roll, offset, margin, snr)``, offset in trading days
    (0 = calendar and prices agree).
    """
    logp = np.log(wide[[1, 2, 3, 4]].where(wide[[1, 2, 3, 4]] > 0))
    spread = pd.DataFrame({k: logp[k + 1] - logp[k] for k in (1, 2, 3)})
    prev = spread.shift(1)
    e_same = (spread[2] - prev[2]).abs() + (spread[3] - prev[3]).abs()
    e_shift = (spread[1] - prev[2]).abs() + (spread[2] - prev[3]).abs()
    score = (e_shift - e_same).to_numpy()
    noise = (spread[2] - prev[2]).abs().rolling(250, min_periods=50).median().to_numpy()
    days = pd.DatetimeIndex(wide.index)
    rows = []
    for roll in expected_rolls:
        i = int(days.searchsorted(roll))
        if i - window < 1 or i + window >= len(days):
            continue
        win = score[i - window : i + window + 1]
        if np.isnan(win).any() or not noise[i] > 0:
            continue
        ordered = np.sort(win)
        margin = float(ordered[1] - ordered[0])
        rows.append((roll, int(np.argmin(win)) - window, margin, margin / float(noise[i])))
    return pd.DataFrame(rows, columns=["expected_roll", "offset", "margin", "snr"])

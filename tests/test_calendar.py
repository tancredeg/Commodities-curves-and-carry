"""NYMEX calendar: business days, last trading days, slot assignment, price-implied rolls."""

from __future__ import annotations

import pandas as pd
import pytest

from energycurves.data import calendar as cal
from energycurves.data.registry import CODES


@pytest.mark.parametrize(
    ("code", "month", "expected"),
    [
        ("CL", "2020-05", "2020-04-21"),  # the negative-price contract
        ("CL", "2009-02", "2009-01-20"),  # 25th on a Sunday, MLK day on the 19th
        ("CL", "2011-03", "2011-02-22"),
        ("CL", "2010-06", "2010-05-20"),
        ("CL", "2022-03", "2022-02-22"),
        ("NG", "2027-04", "2027-03-29"),  # storage-year contract (snapshot)
        ("NG", "2013-12", "2013-11-26"),  # day after Thanksgiving counted (post-2010)
        ("NG", "2010-12", "2010-11-24"),  # 2008-2010 exception
        ("NG", "1994-10", "1994-09-23"),  # 1990s rule: 6 business days
        ("RB", "2020-04", "2020-03-31"),
        ("HO", "2022-05", "2022-04-29"),
    ],
)
def test_known_last_trade_dates(code: str, month: str, expected: str) -> None:
    assert cal.last_trade_date(code, pd.Period(month, freq="M")) == pd.Timestamp(expected)


def test_business_days_holidays_and_exceptions() -> None:
    bd = cal.business_days()
    assert pd.Timestamp("2020-04-10") not in bd  # Good Friday
    assert pd.Timestamp("2018-12-25") not in bd
    assert pd.Timestamp("1998-01-19") not in bd  # first exchange MLK day
    assert pd.Timestamp("1997-01-20") in bd  # MLK not yet an exchange holiday
    assert pd.Timestamp("2012-10-29") in bd  # Sandy: NYSE shut, NYMEX settled
    assert pd.Timestamp("2022-06-20") not in bd  # Juneteenth observed
    assert pd.Timestamp("2001-09-12") not in bd


def test_assign_delivery_months_switches_after_expiry() -> None:
    calendar = cal.expiry_calendar(["CL"], "2020-01", "2020-12")
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-04-21", "2020-04-21", "2020-04-22", "2020-04-22"]),
            "code": "CL",
            "contract": [1, 2, 1, 2],
            "settle": [10.0, 20.0, 13.0, 21.0],
        }
    )
    out = cal.assign_delivery_months(prices, calendar)
    months = out["delivery_month"].dt.strftime("%Y-%m").tolist()
    assert months == ["2020-05", "2020-06", "2020-06", "2020-07"]
    assert out["tau"].iloc[0] == 0.0  # expiry day: contract 1 still the expiring one
    assert (out["tau"] >= 0).all()


def test_calendar_agrees_with_price_implied_rolls(panel: pd.DataFrame) -> None:
    """Calendar vs the roll days revealed by prices (identifiable rolls only).

    A calendar error shows up as a systematic ±1 offset. The remaining
    disagreements were reviewed one by one: the price-based detector fails in
    extreme volatility (CL 2011-02 Libya, 2020-03 Covid, 2022-02 Ukraine; RB
    2020-04) while the rule-based dates match the published CME last trading
    days.
    """
    bd = cal.business_days()
    calendar = cal.expiry_calendar(CODES, "1979-01", "2031-12")
    frames = []
    for code in CODES:
        wide = panel[panel["code"] == code].pivot(index="date", columns="contract", values="settle")
        wide = wide.reindex(bd[(bd >= wide.index.min()) & (bd <= wide.index.max())])
        rolls = cal.roll_dates(calendar, code, pd.DatetimeIndex(wide.index))
        offsets = cal.price_implied_roll_offsets(wide, rolls)
        offsets["code"] = code
        frames.append(offsets)
    res = pd.concat(frames, ignore_index=True)
    identifiable = res[res["snr"] > 8]
    assert len(identifiable) > 400
    assert (identifiable["offset"] == 0).mean() >= 0.98
    one_day = identifiable[identifiable["offset"].abs() == 1]
    reviewed = {("RB", pd.Timestamp("2020-04-01"))}
    assert set(zip(one_day["code"], one_day["expected_roll"], strict=True)) <= reviewed

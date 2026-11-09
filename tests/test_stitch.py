"""Cleaning rules and constant-contract returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energycurves.data import calendar as cal
from energycurves.data.clean import drop_non_business_days, flag_isolated_spikes
from energycurves.data.stitch import excess_return_index, slot_returns


def test_dropped_holiday_rows_are_stale_copies(futures: pd.DataFrame) -> None:
    kept, dropped = drop_non_business_days(futures, cal.business_days())
    assert len(kept) + len(dropped) == len(futures)
    assert 100 < len(dropped) < 200
    prev = (
        futures.sort_values("date")
        .groupby(["code", "contract"])["settle"]
        .shift(1)
        .reindex(futures.index)
    )
    keyed = futures.assign(prev=prev).set_index(["date", "code", "contract"])
    d = keyed.loc[pd.MultiIndex.from_frame(dropped[["date", "code", "contract"]])]
    # Every dropped row repeats the previous settlement, except one row on
    # 2007-01-02 (national day of mourning, exchange closed).
    assert (d["settle"] != d["prev"]).sum() <= 1


def test_spike_filter_flags_known_typos_only(panel: pd.DataFrame, futures: pd.DataFrame) -> None:
    clean, _ = drop_non_business_days(futures, cal.business_days())
    full = cal.assign_delivery_months(
        clean[["date", "code", "contract", "settle"]],
        cal.expiry_calendar(["CL", "NG", "RB", "HO"], "1979-01", "2031-12"),
    )
    flagged = full[flag_isolated_spikes(full)]
    got = set(zip(flagged["code"], flagged["date"].dt.strftime("%Y-%m-%d"), strict=True))
    assert got == {("CL", "2004-07-20"), ("HO", "1999-10-11")}
    assert len(panel) == len(full) - 2


def _synthetic_panel() -> pd.DataFrame:
    """Two contracts in slots 1-2 with a roll between day 2 and day 3."""
    rows = [
        # date, contract(slot), delivery month, price
        ("2020-01-02", 1, "2020-02", 100.0),
        ("2020-01-02", 2, "2020-03", 110.0),
        ("2020-01-03", 1, "2020-02", 101.0),  # last day of the Feb contract
        ("2020-01-03", 2, "2020-03", 111.0),
        ("2020-01-06", 1, "2020-03", 112.0),  # roll: March is now contract 1
        ("2020-01-06", 2, "2020-04", 125.0),
        ("2020-01-07", 1, "2020-03", 113.0),
        ("2020-01-07", 2, "2020-04", 126.0),
    ]
    df = pd.DataFrame(rows, columns=["date", "contract", "delivery_month", "settle"])
    df["date"] = pd.to_datetime(df["date"])
    df["delivery_month"] = pd.to_datetime(df["delivery_month"])
    df["code"] = "XX"
    return df


def test_slot2_index_equals_price_ratio_of_held_contracts() -> None:
    rets = slot_returns(_synthetic_panel(), slot=2).set_index("date")
    # day 2: March 110 -> 111; day 3 (roll): March 111 -> 112, NOT 111 -> 125
    assert np.isclose(rets.loc["2020-01-03", "ret"], 111 / 110 - 1)
    assert np.isclose(rets.loc["2020-01-06", "ret"], 112 / 111 - 1)
    assert bool(rets.loc["2020-01-06", "rolled"])
    # day 4: the position was rolled into April at the close of day 3
    assert np.isclose(rets.loc["2020-01-07", "ret"], 126 / 125 - 1)
    index = excess_return_index(rets.reset_index())
    expected = (111 / 110) * (112 / 111) * (126 / 125)
    assert np.isclose(index["XX"].iloc[-1], expected)


def test_missing_settle_neither_fakes_a_roll_nor_loses_the_return() -> None:
    p = _synthetic_panel()
    gap = p[~((p["date"] == "2020-01-07") & (p["contract"] == 2))]  # April missing on day 4
    extra = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-08", "2020-01-08"]),
            "contract": [1, 2],
            "delivery_month": pd.to_datetime(["2020-03", "2020-04"]),
            "settle": [114.0, 127.0],
            "code": "XX",
        }
    )
    rets = slot_returns(pd.concat([gap, extra], ignore_index=True), slot=2).set_index("date")
    assert np.isnan(rets.loc["2020-01-07", "ret"])  # no April settle on day 4
    assert not bool(rets.loc["2020-01-07", "rolled"])  # ... but no expiry either
    assert np.isclose(rets.loc["2020-01-08", "ret"], 127 / 125 - 1)  # return over the gap
    assert rets["held_month"].loc["2020-01-08"] == pd.Timestamp("2020-04-01")
    assert int(rets["rolled"].sum()) == 1


def test_naive_pct_change_would_include_the_roll_jump() -> None:
    p = _synthetic_panel()
    naive = p[p["contract"] == 2].set_index("date")["settle"].pct_change()
    stitched = slot_returns(p, slot=2).set_index("date")["ret"]
    assert naive.loc["2020-01-06"] > 0.12  # 111 -> 125: the spread, not a return
    assert stitched.loc["2020-01-06"] < 0.01


def test_negative_wti_never_enters_slot2_returns(panel: pd.DataFrame) -> None:
    rets = slot_returns(panel[panel["code"] == "CL"], slot=2)
    window = rets[(rets["date"] >= "2020-04-15") & (rets["date"] <= "2020-04-30")]
    assert window["ret"].notna().all()
    assert (window["ret"] > -1).all()


def test_roll_day_returns_do_not_load_on_the_calendar_spread(panel: pd.DataFrame) -> None:
    """No roll contamination.

    If a roll were dated one day too early, the measured "return" on that day
    would compare two different contracts and equal the front spread:
    ``ln(1 + r_t) ≈ ln(F^1_{t−1} / F^2_{t−1})``, a regression slope of 1.
    With correct stitching the slope is close to 0.
    """
    for code in ("CL", "NG", "RB", "HO"):
        sub = panel[panel["code"] == code]
        wide = sub.pivot(index="date", columns="contract", values="settle")
        logw = np.log(wide.where(wide > 0))
        spread_prev = (logw[1] - logw[2]).shift(1)
        r = slot_returns(sub, slot=2).set_index("date")
        r = r[(r.index >= "2008-01-01") & r["rolled"]]
        frame = pd.DataFrame({"y": np.log1p(r["ret"]), "x": spread_prev.reindex(r.index)}).dropna()
        slope = np.polyfit(frame["x"], frame["y"], 1)[0]
        assert abs(slope) < 0.3, (code, slope)
        assert len(frame) > 180

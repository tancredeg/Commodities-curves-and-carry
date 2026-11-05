"""Module 1 — term structure: slopes, contango/backwardation regimes, roll yield.

For two contracts ``i < j`` observed on the same day, the annualised roll
yield (log slope of the curve per year of maturity) is::

    y(i,j) = (ln F_i − ln F_j) / (τ_j − τ_i),     y > 0  ⇔  backwardation.

Holding a futures contract while the curve stays unchanged, the price drifts
up the curve toward the spot at the rate ``y``: this is the "roll yield"
component of futures returns (Erb & Harvey, 2006; Koijen et al., 2018).
The SQL view ``v_roll_yield`` computes the same quantities; the tests check
that both agree to machine precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BAND = 0.01  # |y12| <= 1 % per year counts as a flat curve


def curve_points(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide view: one row per (code, date) with ``f1..f4``, ``tau1..tau4``, ``m1``."""
    wide = panel.pivot_table(
        index=["code", "date"], columns="contract", values=["settle", "tau"], aggfunc="first"
    )
    out = pd.DataFrame(index=wide.index)
    for k in (1, 2, 3, 4):
        out[f"f{k}"] = wide[("settle", k)]
        out[f"tau{k}"] = wide[("tau", k)]
    m1 = panel[panel["contract"] == 1].set_index(["code", "date"])["delivery_month"]
    out["m1"] = m1.reindex(out.index)
    return out


def _slope(f_i: pd.Series, f_j: pd.Series, tau_i: pd.Series, tau_j: pd.Series) -> pd.Series:
    ok = (f_i > 0) & (f_j > 0)
    num = np.log(f_i.where(ok)) - np.log(f_j.where(ok))
    slope: pd.Series = num / (tau_j - tau_i)
    return slope


def roll_yields(panel: pd.DataFrame) -> pd.DataFrame:
    """Annualised slopes ``y12``, ``y23``, ``y14`` per commodity and day.

    ``y12`` is NaN when contract 1 or 2 is missing or non-positive (WTI on
    2020-04-20): the log of a negative price is undefined.
    """
    pts = curve_points(panel)
    out = pd.DataFrame(index=pts.index)
    out["m1"] = pts["m1"]
    out["y12"] = _slope(pts["f1"], pts["f2"], pts["tau1"], pts["tau2"])
    out["y23"] = _slope(pts["f2"], pts["f3"], pts["tau2"], pts["tau3"])
    out["y14"] = _slope(pts["f1"], pts["f4"], pts["tau1"], pts["tau4"])
    return out.reset_index()


def classify_regime(y: pd.Series, band: float = DEFAULT_BAND) -> pd.Series:
    """'backwardation' if ``y > band``, 'contango' if ``y < −band``, else 'flat'."""
    regime = np.where(y > band, "backwardation", np.where(y < -band, "contango", "flat"))
    return pd.Series(regime, index=y.index).where(y.notna())


def term_structure_summary(
    ry: pd.DataFrame, start: str, end: str, band: float = DEFAULT_BAND
) -> pd.DataFrame:
    """Per commodity over ``[start, end)``: mean/median ``y12``, regime shares, mean ``y14``."""
    sub = ry[(ry["date"] >= start) & (ry["date"] < end) & ry["y12"].notna()].copy()
    sub["regime"] = classify_regime(sub["y12"], band)
    grouped = sub.groupby("code")
    out = pd.DataFrame(
        {
            "n_days": grouped.size(),
            "first_date": grouped["date"].min(),
            "last_date": grouped["date"].max(),
            "mean_y12": grouped["y12"].mean(),
            "median_y12": grouped["y12"].median(),
            "share_backwardation": grouped["regime"].apply(lambda s: (s == "backwardation").mean()),
            "share_contango": grouped["regime"].apply(lambda s: (s == "contango").mean()),
            "share_flat": grouped["regime"].apply(lambda s: (s == "flat").mean()),
            "mean_y14": grouped["y14"].mean(),
        }
    )
    return out.reset_index()


def seasonal_profile(ry: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Mean ``y12`` by calendar month of the front delivery month (``m1``)."""
    sub = ry[(ry["date"] >= start) & (ry["date"] < end) & ry["y12"].notna()].copy()
    sub["front_month"] = sub["m1"].dt.month
    grouped = sub.groupby(["code", "front_month"])["y12"]
    return grouped.agg(n_days="size", mean_y12="mean", sd_y12="std").reset_index()

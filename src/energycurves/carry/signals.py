"""Carry signals: raw front slope and its seasonally adjusted version.

Raw carry of a position in contract 2 (it slides toward contract 1 if the curve
does not move)::

    s_t = y12_t = (ln F1_t − ln F2_t) / (τ2_t − τ1_t)

falling back to ``y23`` when contract 1 is non-positive (WTI on 2020-04-20).

For natural gas and refined products a large part of ``s_t`` is seasonal
(winter premium, summer-grade gasoline). The seasonally adjusted signal
removes the *seasonal shape* estimated on past data only, and keeps the
average carry level::

    s̃_t = s_t − (μ_{m(t)} − μ),
    μ_m = mean{ s_u : u < t, month(m1_u) = m },   μ = mean{ s_u : u < t },

where ``m1`` is the delivery month of contract 1. ``s̃`` is defined once at
least ``min_years`` distinct past years are available for that month.
Subtracting ``μ_{m(t)}`` alone would also remove the long-run carry level
(e.g. the persistent contango of natural gas) and turn the signal into a
"carry anomaly" signal. Koijen et al. (2018) discuss the seasonal component
of commodity carry.
"""

from __future__ import annotations

import pandas as pd

MIN_YEARS = 3


def raw_carry(ry: pd.DataFrame) -> pd.DataFrame:
    """``date, code, m1, raw`` with ``raw = y12`` (``y23`` if ``y12`` is undefined)."""
    out = ry[["date", "code", "m1"]].copy()
    out["raw"] = ry["y12"].fillna(ry["y23"])
    return out.dropna(subset=["raw"]).reset_index(drop=True)


def _prior_mean(values: pd.Series, keys: list[pd.Series]) -> pd.Series:
    """Mean of all earlier rows of the same group (rows already sorted by date)."""
    grp = values.groupby(keys, sort=False)
    prior_sum = grp.cumsum() - values
    prior_n = grp.cumcount()
    mean: pd.Series = prior_sum / prior_n.where(prior_n > 0)
    return mean


def deseasonalize(
    signal: pd.DataFrame, min_years: int = MIN_YEARS, keep_level: bool = True
) -> pd.DataFrame:
    """Add ``seasonal_mean``, ``overall_mean`` and ``deseason`` (past data only).

    Rows are sorted by date within each code; every mean uses *earlier* rows
    only. ``deseason`` is NaN until the earlier rows of the same front month
    cover ``min_years`` distinct calendar years. ``keep_level=False`` gives
    the "carry anomaly" variant ``s − μ_{m(t)}`` (used in the robustness table).
    """
    df = signal.sort_values(["code", "date"]).copy()
    df["front_month"] = df["m1"].dt.month
    df["year"] = df["date"].dt.year
    df["seasonal_mean"] = _prior_mean(df["raw"], [df["code"], df["front_month"]])
    df["overall_mean"] = _prior_mean(df["raw"], [df["code"]])
    # distinct calendar years strictly before the current row's year
    first_of_year = ~df.duplicated(["code", "front_month", "year"])
    years_so_far = first_of_year.astype(int).groupby([df["code"], df["front_month"]]).cumsum()
    past_years = years_so_far - 1
    seasonal_component = df["seasonal_mean"] - (df["overall_mean"] if keep_level else 0.0)
    df["deseason"] = (df["raw"] - seasonal_component).where(past_years >= min_years)
    return df.drop(columns=["year"]).reset_index(drop=True)

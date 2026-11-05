"""Cleaning rules applied to the EIA futures panel before any calculation.

The raw EIA series are versioned untouched in ``data/sample``; the two rules
below are applied when the curve is built (in Python here, and identically in
SQL in ``sql/15_build_curve.sql``; ``tests/test_sql.py`` checks both agree).

1. *Non-trading days.* In 2013-2019 the EIA repeats the previous settlement on
   some exchange holidays (e.g. 2018-12-25 for CL and NG). Rows dated on a day
   that is not a NYMEX business day are dropped.
2. *Isolated spikes on deferred contracts.* A same-contract log return beyond
   ``±threshold`` that is reversed the next day while the rest of the curve is
   quiet is a typing error (e.g. CL contract 4 on 2004-07-20: 35.60 between
   40.38 and 39.26). Only contracts 2-4 are screened: contract 1 can genuinely
   squeeze on its last trading days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPIKE_THRESHOLD = 0.07
SPIKE_REVERSAL = 0.30
SPIKE_QUIET_RATIO = 0.30


def drop_non_business_days(
    prices: pd.DataFrame, business_days: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``prices`` into rows on NYMEX business days and the dropped rest."""
    on_bd = prices["date"].isin(business_days)
    return prices[on_bd].reset_index(drop=True), prices[~on_bd].reset_index(drop=True)


def flag_isolated_spikes(panel: pd.DataFrame, threshold: float = SPIKE_THRESHOLD) -> pd.Series:
    """Boolean mask of isolated, reverting one-day spikes on contracts 2-4.

    ``panel`` needs ``code, delivery_month, date, contract, settle``. With
    ``r_t`` the same-contract log return and ``m_t`` the median absolute return
    of all contracts of the commodity that day, a point is flagged when::

        |r_t| > threshold, |r_{t+1}| > threshold, sign(r_t) ≠ sign(r_{t+1}),
        |r_t + r_{t+1}| < 0.3 |r_t|  and  m_t < 0.3 |r_t|.
    """
    df = panel.sort_values(["code", "delivery_month", "date"])
    logp = np.log(df["settle"].where(df["settle"] > 0))
    grp = logp.groupby([df["code"], df["delivery_month"]])
    ret = grp.diff()
    ret_next = -grp.diff(-1)
    med = ret.abs().groupby([df["code"], df["date"]]).transform("median")
    flag = (
        (df["contract"] >= 2)
        & (ret.abs() > threshold)
        & (ret_next.abs() > threshold)
        & (np.sign(ret) != np.sign(ret_next))
        & ((ret + ret_next).abs() < SPIKE_REVERSAL * ret.abs())
        & (med < SPIKE_QUIET_RATIO * ret.abs())
    )
    out: pd.Series = flag.reindex(panel.index).fillna(False).astype(bool)
    return out

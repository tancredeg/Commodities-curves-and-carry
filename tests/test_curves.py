"""Module 1: roll-yield identities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energycurves import curves


def _panel(prices: list[float], taus: list[float] | None = None) -> pd.DataFrame:
    taus = taus or [0.05, 0.13, 0.21, 0.30]
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2020-01-02"),
            "code": "XX",
            "contract": [1, 2, 3, 4],
            "delivery_month": pd.to_datetime(["2020-02", "2020-03", "2020-04", "2020-05"]),
            "tau": taus,
            "settle": prices,
        }
    )


def test_sign_convention_backwardation_is_positive() -> None:
    back = curves.roll_yields(_panel([60.0, 59.0, 58.0, 57.0])).iloc[0]
    cont = curves.roll_yields(_panel([60.0, 61.0, 62.0, 63.0])).iloc[0]
    assert back["y12"] > 0 and back["y14"] > 0
    assert cont["y12"] < 0 and cont["y14"] < 0


def test_flat_curve_has_zero_roll_yield() -> None:
    ry = curves.roll_yields(_panel([50.0] * 4)).iloc[0]
    assert ry["y12"] == 0 and ry["y23"] == 0 and ry["y14"] == 0


def test_roll_yield_is_scale_invariant_and_matches_formula() -> None:
    prices = [3.0, 3.3, 3.5, 3.2]
    taus = [0.02, 0.10, 0.19, 0.27]
    a = curves.roll_yields(_panel(prices, taus)).iloc[0]
    b = curves.roll_yields(_panel([p * 42.0 for p in prices], taus)).iloc[0]
    assert np.isclose(a["y12"], b["y12"]) and np.isclose(a["y14"], b["y14"])
    assert np.isclose(a["y12"], (np.log(3.0) - np.log(3.3)) / (0.10 - 0.02))


def test_negative_price_gives_missing_not_crash() -> None:
    ry = curves.roll_yields(_panel([-37.63, 20.43, 26.0, 28.0])).iloc[0]
    assert np.isnan(ry["y12"]) and np.isnan(ry["y14"])
    assert np.isfinite(ry["y23"])


def test_regime_dead_band() -> None:
    y = pd.Series([0.05, 0.005, -0.005, -0.05, np.nan])
    assert curves.classify_regime(y, band=0.01).tolist()[:4] == [
        "backwardation",
        "flat",
        "flat",
        "contango",
    ]


def test_summary_on_real_data_is_consistent(panel: pd.DataFrame) -> None:
    ry = curves.roll_yields(panel)
    summary = curves.term_structure_summary(ry, "2008-01-01", "2024-04-06")
    shares = summary[["share_backwardation", "share_contango", "share_flat"]].sum(axis=1)
    assert np.allclose(shares, 1.0)
    assert set(summary["code"]) == {"CL", "NG", "RB", "HO"}
    assert (summary["n_days"] > 4000).all()

"""Regression tests on the frozen sample: key results must not move unnoticed.

Reference values were produced by executing the package on ``data/sample``
(2026-11-05). A failure means a code change altered a published result: either
fix the regression or, if the change is intended, update the value here and
regenerate the README with ``python scripts/run_results.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energycurves import curves
from energycurves.carry import backtest as bt
from energycurves.carry.signals import deseasonalize, raw_carry
from energycurves.data.stitch import slot_returns
from energycurves.spreads.schwartz import calibrate_leg
from energycurves.storage.contract import StorageContract, discount_factors, tbill_continuous_rate
from energycurves.storage.curve import strip_forwards
from energycurves.storage.forward_model import calibrate, calibration_sample
from energycurves.storage.intrinsic import intrinsic_lp
from energycurves.storage.rolling import value_storage

EIA_END = "2024-04-06"
REL = 1e-6  # deterministic computations: only floating-point noise is tolerated
T0 = pd.Timestamp("2026-11-04")


@pytest.fixture(scope="module")
def roll_yields(panel: pd.DataFrame) -> pd.DataFrame:
    return curves.roll_yields(panel)


@pytest.fixture(scope="module")
def storage_inputs(sample_dir: Path) -> tuple[StorageContract, np.ndarray, float]:
    contract = StorageContract()
    strip = pd.read_parquet(sample_dir / "ng_strip_snapshot.parquet")
    rates = pd.read_parquet(sample_dir / "rates_daily.parquet")
    dtb3 = rates[(rates["series_id"] == "DTB3") & (rates["date"] <= T0)].iloc[-1]["value"]
    return contract, strip_forwards(strip, contract), tbill_continuous_rate(float(dtb3))


def test_term_structure_summary(roll_yields: pd.DataFrame) -> None:
    summary = curves.term_structure_summary(roll_yields, "2006-01-01", EIA_END).set_index("code")
    expected = {
        "CL": (-0.08829587771754678, 0.2871569703622393),
        "NG": (-0.2959795789953481, 0.17192982456140352),
        "RB": (0.0348942311628484, 0.6024309392265194),
        "HO": (-0.00923297720045867, 0.3325167037861915),
    }
    for code, (mean_y12, backwardation) in expected.items():
        assert summary.loc[code, "mean_y12"] == pytest.approx(mean_y12, rel=REL)
        assert summary.loc[code, "share_backwardation"] == pytest.approx(backwardation, rel=REL)


def test_carry_sharpe_ratios(panel: pd.DataFrame, roll_yields: pd.DataFrame) -> None:
    config = bt.CarryConfig()
    inputs = bt.prepare_inputs(
        slot_returns(panel, 2), deseasonalize(raw_carry(roll_yields)), config
    )
    start = bt.common_start(inputs, "2006-01-01")
    assert start == "2008-10-01"
    metrics, _, _ = bt.evaluate(inputs, start, EIA_END, config)
    sharpe = metrics.set_index("strategy")["sharpe"]
    expected = {
        "TS-raw": -0.06293746016537974,
        "TS-deseason": 0.08109806390254852,
        "XS-raw": -0.08725850191478246,
        "XS-deseason": 0.25098820487904755,
        "LO": -0.06887471996336665,
    }
    for strategy, value in expected.items():
        assert sharpe[strategy] == pytest.approx(value, rel=REL)


def test_storage_intrinsic_calibration_and_rolling_intrinsic(
    panel: pd.DataFrame, storage_inputs: tuple[StorageContract, np.ndarray, float]
) -> None:
    contract, forwards, rate = storage_inputs
    assert rate == pytest.approx(0.041069264359773545, rel=REL)
    plan = intrinsic_lp(contract, forwards, discount_factors(contract, T0, rate))
    assert plan.value == pytest.approx(1.2245741285678635, rel=REL)
    params = calibrate(calibration_sample(panel, "NG", "2014-04-06", "2024-04-06")).params
    assert params.sigma_s == pytest.approx(0.5911848779624442, rel=1e-4)
    assert params.kappa == pytest.approx(2.494528255045401, rel=1e-4)
    assert params.sigma_l == pytest.approx(0.4196585372583725, rel=1e-4)
    assert params.rho == pytest.approx(-0.32303941526454444, rel=1e-4)
    val, _ = value_storage(
        contract, forwards, T0, rate, params, n_paths=2000, rng=np.random.default_rng(7)
    )
    assert val.rolling_mean == pytest.approx(1.2716049155798566, rel=1e-4)


def test_crack_leg_calibration(panel: pd.DataFrame) -> None:
    expected = {
        "CL": (0.391096063206357, 0.3551507816303242),
        "RB": (0.3805302369844013, 0.4852414799976226),
        "HO": (0.4393948864131483, 1.040180448940641),
    }
    for code, (sigma, kappa) in expected.items():
        leg = calibrate_leg(panel, code)
        assert leg.sigma == pytest.approx(sigma, rel=1e-4)
        assert leg.kappa == pytest.approx(kappa, rel=1e-4)

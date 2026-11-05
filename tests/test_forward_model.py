"""Module 3b: two-factor curve model — covariance, calibration, exact simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energycurves.storage.forward_model import (
    CalibrationSample,
    TwoFactorParams,
    calibrate,
    calibration_sample,
    instantaneous_cov,
    integrated_cov,
    model_moments,
    simulate_curves,
)

TRUE = TwoFactorParams(sigma_s=0.6, kappa=3.0, sigma_l=0.35, rho=-0.3)


def _synthetic_sample(params: TwoFactorParams, n_days: int, seed: int) -> CalibrationSample:
    """Returns drawn from the model with the monthly saw-tooth of slot maturities."""
    rng = np.random.default_rng(seed)
    frac = (np.arange(n_days) % 21) / 21.0  # position inside the expiry month
    taus = (np.arange(4)[None, :] + 1.0 - frac[:, None]) / 12.0
    cov = instantaneous_cov(params, taus[:, :, None], taus[:, None, :]) / 252.0
    eigval, eigvec = np.linalg.eigh(cov)  # rank-2 matrices: no Cholesky
    root = eigvec * np.sqrt(np.clip(eigval, 0.0, None))[:, None, :]
    returns = np.einsum("tij,tj->ti", root, rng.standard_normal((n_days, 4)))
    return CalibrationSample(
        returns=returns, taus=taus, dates=pd.bdate_range("2000-01-03", periods=n_days)
    )


def test_parameter_validation_and_scaling() -> None:
    with pytest.raises(ValueError):
        TwoFactorParams(sigma_s=0.5, kappa=0.0, sigma_l=0.3, rho=0.0)
    with pytest.raises(ValueError):
        TwoFactorParams(sigma_s=0.5, kappa=1.0, sigma_l=0.3, rho=1.5)
    scaled = TRUE.scaled(kappa=2.0, rho=0.0)
    assert scaled.kappa == 6.0 and scaled.rho == 0.0 and scaled.sigma_s == TRUE.sigma_s


def test_instantaneous_covariance_is_psd_and_shows_samuelson_effect() -> None:
    taus = np.linspace(0.0, 2.0, 25)
    cov = instantaneous_cov(TRUE, taus[:, None], taus[None, :])
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-12
    vol = np.sqrt(np.diag(cov))
    assert vol[0] > vol[-1]  # front contracts move more than deferred ones
    # far from expiry only the long factor remains
    assert vol[-1] == pytest.approx(TRUE.sigma_l, rel=0.02)
    positive = TRUE.scaled(rho=0.3)
    vol_pos = np.sqrt(np.diag(instantaneous_cov(positive, taus[:, None], taus[None, :])))
    assert np.all(np.diff(vol_pos) < 0)  # with rho >= 0 the decay is monotone


def test_average_moments_equal_the_mean_of_daily_covariances() -> None:
    taus = np.random.default_rng(0).uniform(0.0, 0.4, (500, 4))
    brute = instantaneous_cov(TRUE, taus[:, :, None], taus[:, None, :]).mean(axis=0)
    assert np.allclose(model_moments(TRUE, taus), brute, rtol=1e-12, atol=1e-15)


def test_integrated_covariance_matches_quadrature_and_freezes_at_expiry() -> None:
    expiries = np.array([0.05, 0.4, 0.9])
    a, b = 0.02, 0.6
    expected = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            upper = min(b, expiries[i], expiries[j])
            if upper <= a:
                continue
            grid = np.linspace(a, upper, 100_001)
            f = instantaneous_cov(TRUE, expiries[i] - grid, expiries[j] - grid)
            expected[i, j] = np.trapezoid(f, grid)
    got = integrated_cov(TRUE, expiries, a, b)
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-9)
    assert np.allclose(integrated_cov(TRUE, expiries, 1.0, 2.0), 0.0)


def test_simulated_futures_are_martingales_and_freeze_after_expiry() -> None:
    F0 = np.array([2.5, 3.0, 4.0, 3.5])
    expiries = np.array([0.1, 0.35, 0.6, 0.85])
    times = np.array([0.05, 0.1, 0.3, 0.6, 0.9])
    paths = simulate_curves(
        TRUE, F0, expiries, times, n_paths=100_000, rng=np.random.default_rng(1)
    )
    final = paths[:, -1, :]
    half = len(final) // 2
    pair_mean = 0.5 * (final[:half] + final[half:])  # antithetic pairs are independent draws
    se = pair_mean.std(axis=0, ddof=1) / np.sqrt(half)
    assert np.all(np.abs(pair_mean.mean(axis=0) - F0) < 3 * se)
    # contract 1 expires at 0.1: frozen afterwards
    assert np.allclose(paths[:, 1, 0], paths[:, -1, 0])
    # contract 3 expires at 0.6: same price at t = 0.6 and t = 0.9
    assert np.allclose(paths[:, 3, 2], paths[:, 4, 2])


def test_simulated_log_increments_have_the_model_covariance() -> None:
    F0 = np.full(3, 3.0)
    expiries = np.array([0.3, 0.6, 1.0])
    paths = simulate_curves(
        TRUE, F0, expiries, np.array([0.25]), n_paths=200_000, rng=np.random.default_rng(4)
    )
    logret = np.log(paths[:, 0, :] / F0)
    expected = integrated_cov(TRUE, expiries, 0.0, 0.25)
    assert np.allclose(np.cov(logret, rowvar=False), expected, rtol=0.02, atol=2e-4)


def test_zero_volatility_leaves_the_curve_unchanged() -> None:
    still = TwoFactorParams(sigma_s=0.0, kappa=1.0, sigma_l=0.0, rho=0.0)
    F0 = np.array([2.0, 3.0])
    out = simulate_curves(
        still,
        F0,
        np.array([0.5, 1.0]),
        np.array([0.2, 0.7]),
        n_paths=10,
        rng=np.random.default_rng(0),
    )
    assert np.allclose(out, F0)


def test_invalid_simulation_requests() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="increasing"):
        simulate_curves(TRUE, np.ones(2), np.ones(2), np.array([0.5, 0.2]), n_paths=4, rng=rng)
    with pytest.raises(ValueError, match="even"):
        simulate_curves(TRUE, np.ones(2), np.ones(2), np.array([0.5]), n_paths=5, rng=rng)


def test_calibration_recovers_known_parameters() -> None:
    result = calibrate(_synthetic_sample(TRUE, 60_000, seed=12))
    p = result.params
    assert p.sigma_s == pytest.approx(TRUE.sigma_s, rel=0.10)
    assert p.sigma_l == pytest.approx(TRUE.sigma_l, rel=0.10)
    assert p.kappa == pytest.approx(TRUE.kappa, rel=0.25)
    assert p.rho == pytest.approx(TRUE.rho, abs=0.10)
    assert result.rmse_corr_scale < 0.02


def test_calibration_sample_skips_roll_days(panel: pd.DataFrame) -> None:
    sample = calibration_sample(panel, "NG", "2014-04-06", "2024-04-06")
    assert sample.returns.shape == sample.taus.shape and sample.returns.shape[1] == 4
    assert np.isfinite(sample.returns).all() and (sample.taus >= 0).all()
    ng = panel[panel["code"] == "NG"]
    first_month = ng[ng["contract"] == 1].set_index("date")["delivery_month"]
    prev_day = pd.Series(first_month.index, index=first_month.index).shift(1)
    rolled_days = first_month.index[first_month != first_month.shift(1)]
    assert not sample.dates.isin(rolled_days).any()
    # spot check: the kept return is the same-contract log price change
    day = sample.dates[100]
    p = ng.set_index(["date", "contract"])["settle"]
    expected = np.log(p[(day, 2)] / p[(prev_day[day], 2)])
    assert sample.returns[100, 1] == pytest.approx(expected)

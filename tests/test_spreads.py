"""Module 4: one-factor legs, Kirk / Margrabe, basket proxy, Monte Carlo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energycurves.spreads.kirk import (
    basket_proxy,
    crack_call_kirk,
    kirk_call,
    kirk_put,
    margrabe_call,
)
from energycurves.spreads.montecarlo import spread_option_mc, terminal_draws
from energycurves.spreads.pricing import SPREADS, price_grid, snapshot
from energycurves.spreads.schwartz import (
    LegParams,
    calibrate_leg,
    fit_leg,
    leg_correlations,
    terminal_log_cov,
)

F1, F2, D = 115.0, 86.0, 0.99
V1, V2, C12 = 0.18, 0.15, 0.8 * 0.18 * 0.15


def test_kirk_with_zero_strike_is_margrabe() -> None:
    m = margrabe_call(F1, F2, v1=V1, v2=V2, c12=C12, D=D)
    k = kirk_call(F1, F2, 0.0, v1=V1, v2=V2, c12=C12, D=D)
    assert k == pytest.approx(m, rel=1e-14)


def test_margrabe_is_exact_under_the_monte_carlo_model() -> None:
    cov = np.array([[V1**2, C12], [C12, V2**2]])
    draws = terminal_draws(np.array([F1, F2]), cov, 400_000, np.random.default_rng(1))
    mc = spread_option_mc(np.array([1.0, -1.0]), draws, 0.0, D)
    assert abs(mc.price - margrabe_call(F1, F2, v1=V1, v2=V2, c12=C12, D=D)) < 3 * mc.se


def test_put_call_parity() -> None:
    """Parity on Monte Carlo payoffs (model-free), and the Kirk put against them.

    kirk_put is *defined* by parity, so C − P = D (F_1 − F_2 − K) on Kirk prices
    would be a tautology; the put is checked against simulated put payoffs instead.
    """
    cov = np.array([[V1**2, C12], [C12, V2**2]])
    draws = terminal_draws(np.array([F1, F2]), cov, 1_000_000, np.random.default_rng(2))
    w = np.array([1.0, -1.0])
    call = spread_option_mc(w, draws, 29.0, D)
    put = spread_option_mc(w, draws, 29.0, D, call=False)
    # pathwise identity, then the martingale property of the draws
    assert call.price - put.price == pytest.approx(D * (draws @ w - 29.0).mean())
    spread_se = D * (draws @ w).std() / np.sqrt(len(draws))
    assert abs(call.price - put.price - D * (F1 - F2 - 29.0)) < 3 * spread_se
    kirk = kirk_put(F1, F2, 29.0, v1=V1, v2=V2, c12=C12, D=D)
    assert abs(kirk - put.price) / put.price < 0.01  # at the money, as for the call
    for K in (-10.0, 0.0, 29.0, 45.0):
        assert kirk_put(F1, F2, K, v1=V1, v2=V2, c12=C12, D=D) >= -1e-12


def test_zero_volatility_gives_the_discounted_intrinsic_value() -> None:
    for K in (-10.0, 0.0, 29.0, 45.0):
        intrinsic = D * max(F1 - F2 - K, 0.0)
        assert kirk_call(F1, F2, K, v1=0.0, v2=0.0, c12=0.0, D=D) == pytest.approx(intrinsic)
    # perfectly correlated legs with equal volatilities: the ratio does not move
    m = margrabe_call(F1, F2, v1=0.2, v2=0.2, c12=0.04, D=D)
    assert m == pytest.approx(D * (F1 - F2))


def test_kirk_is_close_to_monte_carlo_at_the_money() -> None:
    cov = np.array([[V1**2, C12], [C12, V2**2]])
    draws = terminal_draws(np.array([F1, F2]), cov, 1_000_000, np.random.default_rng(3))
    K = F1 - F2
    mc = spread_option_mc(np.array([1.0, -1.0]), draws, K, D)
    kirk = kirk_call(F1, F2, K, v1=V1, v2=V2, c12=C12, D=D)
    assert abs(kirk - mc.price) / mc.price < 0.01
    with pytest.raises(ValueError, match="F2 \\+ K"):
        kirk_call(F1, F2, -F2, v1=V1, v2=V2, c12=C12, D=D)


def test_monte_carlo_error_shrinks_like_one_over_root_n() -> None:
    cov = np.array([[V1**2, C12], [C12, V2**2]])
    w = np.array([1.0, -1.0])
    se = [
        spread_option_mc(
            w, terminal_draws(np.array([F1, F2]), cov, n, np.random.default_rng(4)), 29.0, D
        ).se
        for n in (50_000, 200_000)
    ]
    assert se[0] / se[1] == pytest.approx(2.0, rel=0.1)
    with pytest.raises(ValueError, match="even"):
        terminal_draws(np.array([F1, F2]), cov, 3, np.random.default_rng(0))


def test_basket_proxy_matches_moments() -> None:
    F = np.array([86.0, 2.8, 2.75])
    sd = np.array([0.15, 0.17, 0.16])
    corr = np.array([[1.0, 0.85, 0.8], [0.85, 1.0, 0.75], [0.8, 0.75, 1.0]])
    cov = corr * np.outer(sd, sd)
    w = np.array([0.0, 28.0, 14.0])
    proxy = basket_proxy(w, F, cov, other=0)
    draws = terminal_draws(F, cov, 1_000_000, np.random.default_rng(5))
    basket = draws @ w
    assert proxy.forward == pytest.approx(w @ F)
    assert np.exp(proxy.log_var) * proxy.forward**2 == pytest.approx(np.mean(basket**2), rel=2e-3)
    assert np.exp(proxy.log_cov_other) * proxy.forward * F[0] == pytest.approx(
        np.mean(basket * draws[:, 0]), rel=2e-3
    )
    # a one-asset basket is that asset: proxy = exact moments
    single = basket_proxy(np.array([0.0, 42.0, 0.0]), F, cov, other=0)
    assert single.log_var == pytest.approx(cov[1, 1])
    assert single.log_cov_other == pytest.approx(cov[1, 0])
    one_leg = crack_call_kirk(np.array([0.0, 0.0, 42.0]), F, cov, short=0, K=30.0, D=D)
    direct = kirk_call(42 * F[2], F[0], 30.0, v1=sd[2], v2=sd[0], c12=float(cov[0, 2]), D=D)
    assert one_leg == pytest.approx(direct, rel=1e-12)
    with pytest.raises(ValueError, match="other asset"):
        basket_proxy(np.array([1.0, 28.0, 14.0]), F, cov, other=0)


def test_terminal_covariance_matches_quadrature_and_limits() -> None:
    legs = [LegParams(0.4, 0.5), LegParams(0.35, 1.2)]
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    T, To = np.array([0.30, 0.32]), 0.25
    grid = np.linspace(0.0, To, 200_001)
    expected = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            f = (
                corr[i, j]
                * legs[i].sigma
                * legs[j].sigma
                * np.exp(-legs[i].kappa * (T[i] - grid) - legs[j].kappa * (T[j] - grid))
            )
            expected[i, j] = np.trapezoid(f, grid)
    assert np.allclose(terminal_log_cov(legs, corr, T, To), expected, rtol=1e-8)
    # no mean reversion: plain Black volatility
    flat = [LegParams(0.4, 0.0), LegParams(0.3, 0.0)]
    got = terminal_log_cov(flat, corr, T, To)
    assert got[0, 1] == pytest.approx(0.8 * 0.4 * 0.3 * To)
    with pytest.raises(ValueError, match="expire"):
        terminal_log_cov(legs, corr, np.array([0.2, 0.3]), To)


def test_leg_calibration_recovers_known_parameters() -> None:
    rng = np.random.default_rng(6)
    n = 40_000
    frac = (np.arange(n) % 21) / 21.0
    taus = (np.arange(4)[None, :] + 1.0 - frac[:, None]) / 12.0
    true = LegParams(sigma=0.45, kappa=1.5)
    vol = true.sigma * np.exp(-true.kappa * taus) / np.sqrt(252)
    returns = vol * rng.standard_normal((n, 1))  # one factor: one shock moves every slot
    fit = fit_leg(returns, taus)
    assert fit.sigma == pytest.approx(true.sigma, rel=0.05)
    assert fit.kappa == pytest.approx(true.kappa, rel=0.15)


def test_real_data_calibration_and_grid(panel: pd.DataFrame) -> None:
    legs = [calibrate_leg(panel, code) for code in ("CL", "RB", "HO")]
    assert all(0.1 < leg.sigma < 1.5 and leg.kappa >= 0 for leg in legs)
    corr = leg_correlations(panel, ("CL", "RB", "HO")).to_numpy()
    assert np.allclose(corr, corr.T) and np.all(np.linalg.eigvalsh(corr) > 0)
    snaps = [snapshot(panel, "2024-04-05", "2024-06-01")]
    grid = price_grid(
        snaps, legs, corr, 0.05, (-5.0, 0.0, 5.0), n_paths=20_000, rng=np.random.default_rng(7)
    )
    assert len(grid) == len(SPREADS) * 3
    ho = grid[grid["spread"].str.startswith("HO")].iloc[0]
    cl, ho_price = snaps[0].forwards[0], snaps[0].forwards[2]
    assert ho["forward_spread"] == pytest.approx(42 * ho_price - cl)
    assert (grid["mc_price"] > 0).all() and (grid["kirk_price"] > 0).all()
    with pytest.raises(ValueError, match="no 2030-01"):
        snapshot(panel, "2024-04-05", "2030-01-01")

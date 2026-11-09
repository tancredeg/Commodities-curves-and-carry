"""Two-factor lognormal model of the futures curve: calibration and exact simulation.

Under the pricing measure every futures price is a martingale::

    dF(t,T) / F(t,T) = σ_s e^{−κ(T−t)} dW^s_t + σ_l dW^l_t,   d⟨W^s, W^l⟩ = ρ dt,

with ``T`` the contract's last trading day. The short factor (σ_s, decaying at
rate κ) moves the front of the curve more than the back — the Samuelson
effect — and makes calendar spreads random; the long factor (σ_l) shifts the
whole curve. With a single factor, all contracts are perfectly correlated:
spreads move only through the gap between their volatilities, and the storage
extrinsic value is much smaller (one-factor benchmark in the README); hence two
factors (Schwartz & Smith, 2000, give the same "short-term/long-term" form).

Instantaneous covariance of ``d ln F`` for two contracts with times to expiry
``τ_i, τ_j`` (per year)::

    Σ_ij = σ_s² e^{−κ(τ_i+τ_j)} + σ_l² + ρ σ_s σ_l (e^{−κτ_i} + e^{−κτ_j}).

**Calibration.** Daily same-contract log returns of EIA contracts 1–4 on days
without a roll; ``S`` is their sample covariance ×252 (variance per calendar
year, 252 trading days a year). Because ``τ_k`` cycles through the month, the
model moment is the average of ``Σ(τ_t)`` over the same days; the four
parameters minimise the squared gaps between ``S_ij`` and that average,
each scaled by ``√(S_ii S_jj)`` (a fit on the correlation scale).

**Simulation.** Between two dates ``a < b`` the log increments are exactly
Gaussian: ``ln F_i(b) = ln F_i(a) − ½ C_ii + X_i``, ``X ~ N(0, C)``, with
``C_ij`` the integral of ``Σ_ij`` from ``a`` to ``min(b, T_i, T_j)``
(a contract stops moving at its expiry). No time discretisation error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

TRADING_DAYS = 252
CALIBRATION_WINDOW = ("2014-04-06", "2024-04-06")


@dataclass(frozen=True)
class TwoFactorParams:
    """Short-factor volatility and mean reversion, long-factor volatility, correlation."""

    sigma_s: float
    kappa: float
    sigma_l: float
    rho: float

    def __post_init__(self) -> None:
        if self.sigma_s < 0 or self.sigma_l < 0 or self.kappa <= 0 or not -1 <= self.rho <= 1:
            raise ValueError(f"invalid parameters {self}")

    def scaled(self, **factors: float) -> TwoFactorParams:
        """Copy with some parameters multiplied by a factor (rho is replaced, not scaled)."""
        values = {k: getattr(self, k) for k in ("sigma_s", "kappa", "sigma_l", "rho")}
        for key, factor in factors.items():
            values[key] = factor if key == "rho" else values[key] * factor
        return TwoFactorParams(**values)


def instantaneous_cov(params: TwoFactorParams, tau_i: np.ndarray, tau_j: np.ndarray) -> np.ndarray:
    """``Σ_ij`` per year for times to expiry ``tau_i, tau_j`` (broadcasting)."""
    ei = np.exp(-params.kappa * np.asarray(tau_i))
    ej = np.exp(-params.kappa * np.asarray(tau_j))
    ss, sl, rho = params.sigma_s, params.sigma_l, params.rho
    out: np.ndarray = ss * ss * ei * ej + sl * sl + rho * ss * sl * (ei + ej)
    return out


def model_moments(params: TwoFactorParams, taus: np.ndarray) -> np.ndarray:
    """Average of ``Σ(τ_t)`` over the rows of ``taus`` (shape ``(T, K)``): a ``K × K`` matrix.

    Uses ``mean_t e^{−κ(τ_i+τ_j)} = (EᵀE)_ij / T`` and ``mean_t e^{−κτ_i}`` with
    ``E_ti = e^{−κ τ_ti}``, which avoids building the ``T × K × K`` array.
    """
    e = np.exp(-params.kappa * taus)
    outer = e.T @ e / len(e)
    mean_e = e.mean(axis=0)
    ss, sl, rho = params.sigma_s, params.sigma_l, params.rho
    out: np.ndarray = (
        ss * ss * outer + sl * sl + rho * ss * sl * (mean_e[:, None] + mean_e[None, :])
    )
    return out


@dataclass(frozen=True)
class CalibrationSample:
    """Same-contract daily log returns of slots 1..K and their times to expiry."""

    returns: np.ndarray  # (T, K)
    taus: np.ndarray  # (T, K), time to expiry at the start of each return (years)
    dates: pd.DatetimeIndex


def calibration_sample(
    panel: pd.DataFrame,
    code: str = "NG",
    start: str = CALIBRATION_WINDOW[0],
    end: str = CALIBRATION_WINDOW[1],
    contracts: tuple[int, ...] = (1, 2, 3, 4),
) -> CalibrationSample:
    """Returns ``ln F_t(D) − ln F_{t−1}(D)`` of the contracts in each slot on non-roll days.

    A day is kept when every slot holds the same delivery month as the previous
    trading day (no expiry in between) and all prices are positive.
    """
    sub = panel[(panel["code"] == code) & (panel["date"] >= start) & (panel["date"] < end)]
    settle = sub.pivot(index="date", columns="contract", values="settle")[list(contracts)]
    month = sub.pivot(index="date", columns="contract", values="delivery_month")[list(contracts)]
    tau = sub.pivot(index="date", columns="contract", values="tau")[list(contracts)]
    positive = settle.where(settle > 0).to_numpy(dtype=float)
    logp = pd.DataFrame(np.log(positive), index=settle.index, columns=settle.columns)
    same = (month == month.shift(1)).all(axis=1)
    ret = logp.diff()
    keep = same & ret.notna().all(axis=1) & tau.shift(1).notna().all(axis=1)
    return CalibrationSample(
        returns=ret[keep].to_numpy(),
        taus=tau.shift(1)[keep].to_numpy(dtype=float),
        dates=pd.DatetimeIndex(ret.index[keep]),
    )


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted parameters, empirical and fitted moment matrices (per year), fit error."""

    params: TwoFactorParams
    empirical: np.ndarray
    fitted: np.ndarray
    rmse_corr_scale: float
    n_days: int


def calibrate(sample: CalibrationSample) -> CalibrationResult:
    """Least-squares fit of the two-factor covariance to the sample moments (multi-start)."""
    emp = np.cov(sample.returns, rowvar=False) * TRADING_DAYS
    scale = np.sqrt(np.outer(np.diag(emp), np.diag(emp)))
    iu = np.triu_indices(emp.shape[0])

    def residuals(x: np.ndarray) -> np.ndarray:
        params = TwoFactorParams(sigma_s=x[0], kappa=x[1], sigma_l=x[2], rho=x[3])
        out: np.ndarray = ((model_moments(params, sample.taus) - emp) / scale)[iu]
        return out

    vol_front = float(np.sqrt(emp[0, 0]))
    lower, upper = [1e-4, 1e-3, 0.0, -0.999], [5.0, 50.0, 3.0, 0.999]
    best = None
    for kappa0 in (0.5, 2.0, 8.0):
        for rho0 in (-0.5, 0.0, 0.5):
            x0 = [vol_front, kappa0, 0.5 * vol_front, rho0]
            fit = least_squares(residuals, x0, bounds=(lower, upper), method="trf", xtol=1e-12)
            if best is None or fit.cost < best.cost:
                best = fit
    assert best is not None
    params = TwoFactorParams(*map(float, best.x))
    fitted = model_moments(params, sample.taus)
    rmse = float(np.sqrt(np.mean(best.fun**2)))
    return CalibrationResult(
        params=params, empirical=emp, fitted=fitted, rmse_corr_scale=rmse, n_days=len(sample.dates)
    )


def integrated_cov(params: TwoFactorParams, expiries: np.ndarray, a: float, b: float) -> np.ndarray:
    """``C_ij = ∫_a^{u_ij} Σ_ij(T_i − t, T_j − t) dt``, ``u_ij = min(b, T_i, T_j)`` (years)."""
    T = np.asarray(expiries, dtype=float)
    u = np.minimum(b, np.minimum.outer(T, T))
    length = np.clip(u - a, 0.0, None)
    u = a + length  # u = a where the pair is already frozen
    k, ss, sl, rho = params.kappa, params.sigma_s, params.sigma_l, params.rho
    ti, tj = T[:, None], T[None, :]
    # ∫ e^{−κ(T_i−t) − κ(T_j−t)} dt and ∫ e^{−κ(T_k−t)} dt over [a, u]
    short = (np.exp(-k * (ti - u) - k * (tj - u)) - np.exp(-k * (ti - a) - k * (tj - a))) / (2 * k)
    lin_i = (np.exp(-k * (ti - u)) - np.exp(-k * (ti - a))) / k
    lin_j = (np.exp(-k * (tj - u)) - np.exp(-k * (tj - a))) / k
    out: np.ndarray = ss * ss * short + sl * sl * length + rho * ss * sl * (lin_i + lin_j)
    return out


def simulate_curves(
    params: TwoFactorParams,
    forwards: np.ndarray,
    expiries: np.ndarray,
    times: np.ndarray,
    *,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool = True,
) -> np.ndarray:
    """Futures prices of ``M`` contracts at each of ``J`` future ``times`` (years from today).

    Returns an array ``(n_paths, J, M)``; ``expiries`` are the contracts' last
    trading days in years from today. Antithetic pairs halve the noise of
    linear payoffs; ``n_paths`` must then be even.
    """
    F0 = np.asarray(forwards, dtype=float)
    times = np.asarray(times, dtype=float)
    if np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError("times must be increasing and non-negative")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even with antithetic sampling")
    M = len(F0)
    logf = np.tile(np.log(F0), (n_paths, 1))
    out = np.empty((n_paths, len(times), M))
    prev = 0.0
    for j, t in enumerate(times):
        if t > prev:
            cov = integrated_cov(params, expiries, prev, t)
            eigval, eigvec = np.linalg.eigh(cov)
            root = eigvec * np.sqrt(np.clip(eigval, 0.0, None))
            half = n_paths // 2 if antithetic else n_paths
            z = rng.standard_normal((half, M))
            if antithetic:
                z = np.vstack([z, -z])
            logf = logf - 0.5 * np.diag(cov) + z @ root.T
        out[:, j, :] = np.exp(logf)
        prev = t
    return out

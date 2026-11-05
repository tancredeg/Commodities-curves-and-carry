"""One-factor mean-reverting (Schwartz) futures model per leg, and its calibration.

In the one-factor model of Schwartz (1997) the log spot price is an
Ornstein–Uhlenbeck process; the implied futures dynamics are::

    dF(t,T) / F(t,T) = σ e^{−κ(T−t)} dW_t,

so a contract's volatility rises as it approaches expiry (Samuelson effect)
and ``κ`` measures how fast it decays along the curve. Legs ``i, j`` (crude,
gasoline, heating oil) are driven by correlated Brownian motions,
``d⟨W_i, W_j⟩ = ρ_ij dt``.

**Calibration.** ``σ`` and ``κ`` per leg: the sample variance of daily
same-contract log returns of slots 1–4 (non-roll days, ×252) is matched to the
average of ``σ² e^{−2κ τ_t}`` over the same days (least squares on logs).
``ρ_ij``: correlation of the daily returns of contract 2 of each leg on days
where no leg rolls; in this model the correlation of two legs' returns is
exactly ``ρ_ij`` whatever the maturities.

**Terminal distribution.** For an option expiring at ``T_o`` on futures
expiring at ``T_i ≥ T_o``, ``ln F_i(T_o)`` is Gaussian with mean
``ln F_i(0) − ½ C_ii`` and covariance::

    C_ij = ρ_ij σ_i σ_j e^{−κ_i (T_i − T_o) − κ_j (T_j − T_o)}
           · (1 − e^{−(κ_i+κ_j) T_o}) / (κ_i + κ_j).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

TRADING_DAYS = 252
CALIBRATION_WINDOW = ("2021-04-06", "2024-04-06")


@dataclass(frozen=True)
class LegParams:
    """Volatility scale and mean-reversion speed of one leg."""

    sigma: float
    kappa: float


def _same_contract_returns(
    panel: pd.DataFrame, code: str, start: str, end: str, contracts: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily log returns of slots ``contracts`` on non-roll days, and ``τ`` at the start."""
    sub = panel[(panel["code"] == code) & (panel["date"] >= start) & (panel["date"] < end)]
    settle = sub.pivot(index="date", columns="contract", values="settle")[list(contracts)]
    month = sub.pivot(index="date", columns="contract", values="delivery_month")[list(contracts)]
    tau = sub.pivot(index="date", columns="contract", values="tau")[list(contracts)]
    positive = settle.where(settle > 0).to_numpy(dtype=float)
    logp = pd.DataFrame(np.log(positive), index=settle.index, columns=settle.columns)
    ret = logp.diff().where(month == month.shift(1))
    return ret, tau.shift(1)


def _decay_integral(rate: float, horizon: float) -> float:
    """``(1 − e^{−rate·T}) / rate``, equal to ``T`` in the limit ``rate → 0``."""
    if rate * horizon < 1e-10:
        return horizon
    return float(-np.expm1(-rate * horizon) / rate)


def calibrate_leg(
    panel: pd.DataFrame,
    code: str,
    start: str = CALIBRATION_WINDOW[0],
    end: str = CALIBRATION_WINDOW[1],
) -> LegParams:
    """Fit ``σ, κ`` so that ``mean_t σ² e^{−2κτ_tk}`` matches the variance of each slot ``k``."""
    ret, tau = _same_contract_returns(panel, code, start, end, (1, 2, 3, 4))
    return fit_leg(ret.to_numpy(dtype=float), tau.to_numpy(dtype=float))


def fit_leg(returns: np.ndarray, taus: np.ndarray) -> LegParams:
    """Least squares on log variances; ``returns, taus`` are ``(T, K)`` (NaN = no return).

    Model variance of slot ``k``: the average of ``σ² e^{−2κτ}`` over the days
    on which slot ``k`` has a return (``×252`` annualisation on the data side).
    """
    variances, tau_lists = [], []
    for k in range(returns.shape[1]):
        ok = np.isfinite(returns[:, k]) & np.isfinite(taus[:, k])
        variances.append(float(np.var(returns[ok, k], ddof=1)) * TRADING_DAYS)
        tau_lists.append(taus[ok, k])
    log_var = np.log(variances)

    def residuals(x: np.ndarray) -> np.ndarray:
        sigma, kappa = x
        model = [sigma**2 * np.mean(np.exp(-2.0 * kappa * t)) for t in tau_lists]
        out: np.ndarray = np.log(model) - log_var
        return out

    fit = least_squares(
        residuals, [np.sqrt(variances[0]), 1.0], bounds=([1e-4, 0.0], [5.0, 50.0]), xtol=1e-12
    )
    return LegParams(sigma=float(fit.x[0]), kappa=float(fit.x[1]))


def leg_correlations(
    panel: pd.DataFrame,
    codes: tuple[str, ...],
    start: str = CALIBRATION_WINDOW[0],
    end: str = CALIBRATION_WINDOW[1],
    contract: int = 2,
) -> pd.DataFrame:
    """Correlation of same-contract daily returns of ``contract`` across legs (no-roll days)."""
    cols = {
        code: _same_contract_returns(panel, code, start, end, (contract,))[0][contract]
        for code in codes
    }
    joint = pd.DataFrame(cols).dropna()
    return joint.corr()


def terminal_log_cov(
    legs: list[LegParams], corr: np.ndarray, futures_expiries: np.ndarray, option_expiry: float
) -> np.ndarray:
    """Covariance of ``ln F_i(T_o)`` across legs (formula in the module docstring)."""
    n = len(legs)
    T = np.asarray(futures_expiries, dtype=float)
    if np.any(option_expiry - T > 1e-12):
        raise ValueError("every futures contract must expire on or after the option")
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            ki, kj = legs[i].kappa, legs[j].kappa
            damp = np.exp(-ki * (T[i] - option_expiry) - kj * (T[j] - option_expiry))
            out[i, j] = (
                corr[i, j]
                * legs[i].sigma
                * legs[j].sigma
                * damp
                * _decay_integral(ki + kj, option_expiry)
            )
    return out

"""Monte Carlo benchmark for spread options under the same joint-lognormal model.

Each leg's futures price at the option expiry is lognormal with the covariance
of :func:`energycurves.spreads.schwartz.terminal_log_cov`, so the terminal
values are drawn directly — no time stepping, hence no discretisation error::

    X_k = F_k exp(−½ C_kk + Z_k),   Z ~ N(0, C).

The payoff is ``(Σ_k w_k X_k − K)^+`` where the weights carry the sign of each
leg (e.g. ``w = (−1, 2·42/3, 42/3)`` for a 3-2-1 crack in USD/bbl with crude
first). Antithetic pairs reduce variance; the standard error uses pair means.
Because Kirk and Monte Carlo share the model, their gap is the error of the
approximation alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCPrice:
    """Discounted Monte Carlo price, its standard error and the number of draws."""

    price: float
    se: float
    n_paths: int


def terminal_draws(
    forwards: np.ndarray, log_cov: np.ndarray, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_paths, n_legs)`` terminal prices with antithetic pairs (``n_paths`` even)."""
    if n_paths % 2:
        raise ValueError("n_paths must be even (antithetic pairs)")
    F = np.asarray(forwards, dtype=float)
    eigval, eigvec = np.linalg.eigh(log_cov)
    root = eigvec * np.sqrt(np.clip(eigval, 0.0, None))
    z = rng.standard_normal((n_paths // 2, len(F)))
    z = np.vstack([z, -z]) @ root.T
    out: np.ndarray = F * np.exp(-0.5 * np.diag(log_cov) + z)
    return out


def spread_option_mc(
    weights: np.ndarray,
    draws: np.ndarray,
    K: float,
    D: float,
    *,
    call: bool = True,
) -> MCPrice:
    """Price ``(w·X − K)^+`` (call) or ``(K − w·X)^+`` (put) on pre-drawn terminal prices."""
    spread = draws @ np.asarray(weights, dtype=float) - K
    payoff = np.maximum(spread if call else -spread, 0.0)
    half = len(payoff) // 2
    pairs = 0.5 * (payoff[:half] + payoff[half:])
    return MCPrice(
        price=float(D * payoff.mean()),
        se=float(D * pairs.std(ddof=1) / np.sqrt(half)),
        n_paths=len(payoff),
    )

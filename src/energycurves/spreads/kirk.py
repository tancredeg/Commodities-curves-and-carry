"""Closed-form spread-option approximations: Margrabe, Kirk, and a basket extension.

Two lognormal assets ``X_1, X_2`` with forwards ``F_1, F_2``, total log
variances ``v_1², v_2²`` and log covariance ``c_12`` over the option's life.
A call on the spread pays ``(X_1 − X_2 − K)^+``.

* **Margrabe (1978)**, exact for ``K = 0`` (exchange option)::

      C = D [F_1 N(d_1) − F_2 N(d_2)],  v² = v_1² − 2 c_12 + v_2²,
      d_1 = (ln(F_1/F_2) + ½ v²) / v,   d_2 = d_1 − v.

* **Kirk (1995)** treats ``X_2 + K`` as lognormal with the volatility of
  ``X_2`` scaled by ``w = F_2 / (F_2 + K)``::

      C ≈ D [F_1 N(d_1) − (F_2 + K) N(d_2)],
      v_K² = v_1² − 2 w c_12 + w² v_2²,
      d_1 = (ln(F_1/(F_2+K)) + ½ v_K²) / v_K,  d_2 = d_1 − v_K.

  With ``K = 0`` it is Margrabe's formula exactly. It is accurate near the
  money and degrades for large strikes (Carmona & Durrleman, 2003).

* **3-2-1 crack spread.** Kirk handles two assets only. The product leg
  ``B = Σ a_k X_k`` (gasoline and heating oil) is replaced by a lognormal
  variable with the same mean and second moment, and the same cross moment
  with crude; Kirk is then applied to ``(B, crude)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

VOL_FLOOR = 1e-12  # below this total volatility the option is worth its intrinsic value


def _exchange_value(F1: float, F2: float, v: float, D: float) -> float:
    """``D E[(X_1 − X_2)^+]`` when ``ln(X_1/X_2)`` has total standard deviation ``v``.

    For ``v → 0`` the formula tends to ``D (F_1 − F_2)^+``; that limit is used
    below ``VOL_FLOOR`` instead of dividing by zero.
    """
    if v <= VOL_FLOOR:
        return float(D * max(F1 - F2, 0.0))
    d1 = (np.log(F1 / F2) + 0.5 * v * v) / v
    return float(D * (F1 * norm.cdf(d1) - F2 * norm.cdf(d1 - v)))


def margrabe_call(F1: float, F2: float, *, v1: float, v2: float, c12: float, D: float) -> float:
    """Exchange option ``(X_1 − X_2)^+``: exact under joint lognormality."""
    return _exchange_value(F1, F2, float(np.sqrt(max(v1 * v1 - 2.0 * c12 + v2 * v2, 0.0))), D)


def kirk_call(
    F1: float, F2: float, K: float, *, v1: float, v2: float, c12: float, D: float
) -> float:
    """Kirk's approximation of ``(X_1 − X_2 − K)^+`` (requires ``F_2 + K > 0``).

    ``v1, v2`` are total log standard deviations (σ √T), ``c12`` the total log covariance.
    """
    base = F2 + K
    if base <= 0:
        raise ValueError("Kirk's approximation needs F2 + K > 0")
    w = F2 / base
    vk = float(np.sqrt(max(v1 * v1 - 2.0 * w * c12 + w * w * v2 * v2, 0.0)))
    return _exchange_value(F1, base, vk, D)


def kirk_put(
    F1: float, F2: float, K: float, *, v1: float, v2: float, c12: float, D: float
) -> float:
    """Put on the spread from put–call parity: ``P = C − D (F_1 − F_2 − K)``."""
    return kirk_call(F1, F2, K, v1=v1, v2=v2, c12=c12, D=D) - D * (F1 - F2 - K)


@dataclass(frozen=True)
class LognormalProxy:
    """Moment-matched lognormal stand-in for a basket, and its log covariance with another asset."""

    forward: float
    log_var: float
    log_cov_other: float


def basket_proxy(
    weights: np.ndarray, forwards: np.ndarray, log_cov: np.ndarray, other: int
) -> LognormalProxy:
    """Lognormal proxy of ``B = Σ_k w_k X_k`` (``w_other = 0``).

    Matches ``E[B]``, ``E[B²]`` and ``E[B X_other]``.

    For jointly lognormal ``X`` with ``E[X_k] = F_k``: ``E[X_i X_j] = F_i F_j e^{C_ij}``, so

        v_B² = ln(E[B²] / E[B]²),   c_B,o = ln(E[B X_o] / (E[B] F_o)).
    """
    w = np.asarray(weights, dtype=float)
    F = np.asarray(forwards, dtype=float)
    if w[other] != 0:
        raise ValueError("the basket must not contain the other asset")
    a = w * F
    mean = float(a.sum())
    second = float(a @ np.exp(log_cov) @ a)
    cross = float((a * np.exp(log_cov[:, other])).sum())
    return LognormalProxy(
        forward=mean,
        log_var=float(np.log(second / mean**2)),
        log_cov_other=float(np.log(cross / mean)),
    )


def crack_call_kirk(
    weights: np.ndarray,
    forwards: np.ndarray,
    log_cov: np.ndarray,
    *,
    short: int,
    K: float,
    D: float,
) -> float:
    """Call on ``Σ_k w_k X_k − X_short − K``: basket proxy for the long leg, then Kirk."""
    proxy = basket_proxy(weights, forwards, log_cov, short)
    return kirk_call(
        proxy.forward,
        float(forwards[short]),
        K,
        v1=float(np.sqrt(proxy.log_var)),
        v2=float(np.sqrt(log_cov[short, short])),
        c12=proxy.log_cov_other,
        D=D,
    )

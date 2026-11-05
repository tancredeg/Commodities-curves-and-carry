"""Performance metrics of daily excess-return series.

All series are futures excess returns (fully collateralised positions), so
the Sharpe ratio is computed directly on them, with 252 trading days a year:

    mean = 252 · avg(r),  vol = √252 · sd(r),  SR = mean / vol,
    t ≈ SR · √years   (i.i.d. approximation),
    MDD = min_t (NAV_t / max_{s≤t} NAV_s − 1),  NAV_t = Π (1 + r).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough loss of the compounded NAV (a negative number)."""
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def performance(returns: pd.Series) -> dict[str, float]:
    """Annualised return, volatility, Sharpe, t-stat, max drawdown, monthly hit rate."""
    r = returns.dropna()
    years = len(r) / TRADING_DAYS
    mean = float(r.mean() * TRADING_DAYS)
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = mean / vol if vol > 0 else float("nan")
    months = pd.DatetimeIndex(r.index).to_period("M")
    monthly = (1.0 + r).groupby(months).prod() - 1.0
    return {
        "ann_return": mean,
        "ann_vol": vol,
        "sharpe": sharpe,
        "t_stat": sharpe * np.sqrt(years),
        "max_drawdown": max_drawdown(r),
        "hit_rate_monthly": float((monthly > 0).mean()),
        "years": years,
    }

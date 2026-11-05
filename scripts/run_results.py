"""Regenerate every table and figure of the README from the frozen sample.

Usage::

    python scripts/run_results.py           # write docs/results, docs/figures, README blocks
    python scripts/run_results.py --check   # recompute; exit 1 if a README block would change

Every number shown in ``README.md`` and ``docs/DATA.md`` sits between
``<!-- BEGIN:name -->`` / ``<!-- END:name -->`` markers and is written here, so
the documents cannot drift from the code. Full-precision tables go to
``docs/results/*.csv``; figures to ``docs/figures/*.png``. Random draws use
fixed seeds; the whole run takes well under a minute.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from energycurves import curves
from energycurves.carry import backtest as bt
from energycurves.carry.signals import deseasonalize, raw_carry
from energycurves.data.registry import NG_STRIP_SNAPSHOT_DATE
from energycurves.data.stitch import build_curve_panel, slot_returns
from energycurves.db import build_database, run_query
from energycurves.spreads.pricing import LEGS, price_grid, snapshot
from energycurves.spreads.schwartz import calibrate_leg, leg_correlations
from energycurves.storage.contract import StorageContract, discount_factors, tbill_continuous_rate
from energycurves.storage.curve import chained_seasonal_curve, seasonal_log_spreads, strip_forwards
from energycurves.storage.forward_model import TwoFactorParams, calibrate, calibration_sample
from energycurves.storage.intrinsic import IntrinsicPlan, intrinsic_lp
from energycurves.storage.rolling import (
    RollingResult,
    StorageValuation,
    sensitivity,
    value_storage,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"
DOCUMENTS = (ROOT / "README.md", ROOT / "docs" / "DATA.md")

SEED = 20261104
CODES_ORDER = ("CL", "NG", "RB", "HO")
NAMES = {"CL": "WTI crude", "NG": "Henry Hub gas", "RB": "RBOB gasoline", "HO": "Heating oil"}
EIA_END = "2024-04-06"  # day after the last EIA futures settlement (2024-04-05)
M1_PERIODS = (("2006-01-01", EIA_END), ("2006-01-01", "2015-01-01"), ("2015-01-01", EIA_END))
CARRY_NOT_BEFORE = "2006-01-01"
STORAGE_PATHS = 10_000
SPREAD_PATHS = 2_000_000
SPREAD_VALUATION = "2024-04-05"
SPREAD_MONTHS = ("2024-06-01", "2024-07-01", "2024-08-01")
SPREAD_OFFSETS = (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)

# Chart chrome (light surface) and categorical slots, validated with the dataviz checks.
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
FIG_META = {"Author": "Tancrède Geli", "Creator": "energycurves scripts/run_results.py"}


@dataclass
class Output:
    """Markdown blocks for the documents, CSV tables and figure writers."""

    blocks: dict[str, str] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Callable[[Path], None]] = field(default_factory=dict)


# ---------------------------------------------------------------- formatting


def pct(x: float, digits: int = 1, sign: bool = False) -> str:
    return f"{100 * x:+.{digits}f}%" if sign else f"{100 * x:.{digits}f}%"


def num(x: float, digits: int = 2, sign: bool = False) -> str:
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def md_table(frame: pd.DataFrame, align: str | None = None) -> str:
    """GitHub markdown table from already formatted string cells."""
    cols = [str(c) for c in frame.columns]
    align = align or "l" + "r" * (len(cols) - 1)
    rule = ["---:" if a == "r" else ":---" if a == "l" else ":---:" for a in align]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(rule) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.to_numpy()]
    return "\n".join(lines)


def period_label(start: str, end: str) -> str:
    last = pd.Timestamp(end) - pd.Timedelta(days=1)
    return f"{start[:4]}–{last.year}"


# ---------------------------------------------------------------- charts


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _figure(nrows: int, ncols: int, size: tuple[float, float], **kw: object) -> tuple:
    fig, axes = plt.subplots(nrows, ncols, figsize=size, facecolor=SURFACE, **kw)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _save(fig: plt.Figure, path: Path, title: str) -> None:
    fig.savefig(path, dpi=150, facecolor=SURFACE, metadata={**FIG_META, "Title": title})
    plt.close(fig)


# ---------------------------------------------------------------- module 1


def module1(panel: pd.DataFrame, out: Output) -> pd.DataFrame:
    ry = curves.roll_yields(panel)
    rows = []
    for start, end in M1_PERIODS:
        summary = curves.term_structure_summary(ry, start, end)
        summary.insert(0, "period", period_label(start, end))
        rows.append(summary)
    table = pd.concat(rows, ignore_index=True)
    out.tables["term_structure"] = table
    seasonal = curves.seasonal_profile(ry, "2006-01-01", EIA_END)
    out.tables["term_structure_seasonal"] = seasonal

    view = table.set_index(["period", "code"])
    lines = []
    for code in CODES_ORDER:
        cells = [f"{code} ({NAMES[code]})"]
        for start, end in M1_PERIODS:
            r = view.loc[(period_label(start, end), code)]
            cells += [pct(r["mean_y12"], sign=True), pct(r["share_backwardation"], 0)]
        lines.append(cells)
    cols = ["Contract"]
    for start, end in M1_PERIODS:
        label = period_label(start, end)
        cols += [f"mean y12 {label}", f"backwardation {label}"]
    out.blocks["m1-table"] = md_table(pd.DataFrame(lines, columns=cols))

    season = seasonal.pivot(index="front_month", columns="code", values="mean_y12")
    month_names = [pd.Timestamp(2000, m, 1).strftime("%b") for m in season.index]
    srows = [
        [code] + [pct(season.loc[m, code], 0, sign=True) for m in season.index]
        for code in ("NG", "RB", "HO")
    ]
    out.blocks["m1-seasonal"] = md_table(
        pd.DataFrame(srows, columns=["Front month →", *month_names])
    )

    def fig(path: Path) -> None:
        fg, axes = _figure(2, 2, (10, 6), sharex=True)
        for ax, code in zip(axes.ravel(), CODES_ORDER, strict=True):
            s = ry[ry["code"] == code].set_index("date")["y12"].dropna()
            smooth = s.rolling(252, min_periods=200).mean() * 100
            ax.axhline(0, color=AXIS, linewidth=0.8)
            ax.plot(smooth.index, smooth.to_numpy(), color=SLOTS[0], linewidth=1.5)
            ax.set_title(f"{code} — {NAMES[code]}", fontsize=10, loc="left")
            ax.set_ylabel("% per year")
        fg.suptitle(
            "Front roll yield y12, 12-month moving average (> 0 = backwardation)",
            fontsize=11,
            color=INK,
            x=0.01,
            ha="left",
        )
        fg.tight_layout()
        _save(fg, path, "Front roll yield by contract")

    out.figures["m1_roll_yield.png"] = fig
    return ry


# ---------------------------------------------------------------- module 2


def module2(panel: pd.DataFrame, ry: pd.DataFrame, out: Output) -> dict[str, float]:
    signal_raw = raw_carry(ry)
    signals = deseasonalize(signal_raw)
    anomaly = deseasonalize(signal_raw, keep_level=False)
    slot2 = slot_returns(panel, 2)
    config = bt.CarryConfig()
    inputs = bt.prepare_inputs(slot2, signals, config)
    start = bt.common_start(inputs, CARRY_NOT_BEFORE)
    metrics, costs, navs = bt.evaluate(inputs, start, EIA_END, config)
    robust = bt.robustness(slot2, {"shape": signals, "anomaly": anomaly}, start, EIA_END)
    out.tables["carry_metrics"] = metrics
    out.tables["carry_costs"] = costs
    out.tables["carry_robustness"] = robust

    labels = {
        "TS-raw": "TS, raw slope",
        "TS-deseason": "TS, seasonally adjusted",
        "XS-raw": "XS, raw slope",
        "XS-deseason": "XS, seasonally adjusted",
        "LO": "Long only, equal risk",
    }
    rows = [
        [
            labels[r.strategy],
            pct(r.ann_return, 1, sign=True),
            pct(r.ann_vol, 1),
            num(r.sharpe, 2, sign=True),
            num(r.t_stat, 2, sign=True),
            pct(r.max_drawdown, 0),
            num(r.turnover, 1),
        ]
        for r in metrics.itertuples()
    ]
    first, last = metrics["start"].iloc[0], metrics["end"].iloc[0]
    years = metrics["years"].iloc[0]
    out.blocks["m2-table"] = (
        md_table(
            pd.DataFrame(
                rows,
                columns=[
                    "Strategy",
                    "Return / yr",
                    "Vol / yr",
                    "Sharpe",
                    "t-stat",
                    "Max DD",
                    "Turnover / yr",
                ],
            )
        )
        + f"\n\nDaily excess returns, {first} to {last} ({years:.1f} years), no trading costs."
    )
    piv = costs.pivot(index="strategy", columns="cost_bps", values="sharpe").loc[list(labels)]
    crow = [[labels[s], *(num(v, 2, sign=True) for v in piv.loc[s])] for s in piv.index]
    out.blocks["m2-costs"] = md_table(
        pd.DataFrame(
            crow, columns=["Sharpe at cost (bp of notional) →", *(f"{c:g}" for c in piv.columns)]
        )
    )
    rrow = robust.pivot(index="variant", columns="strategy", values="sharpe")
    rrow = rrow.loc[robust["variant"].drop_duplicates()]
    out.blocks["m2-robustness"] = md_table(
        pd.DataFrame(
            [
                [
                    v,
                    num(rrow.loc[v, "TS-deseason"], 2, sign=True),
                    num(rrow.loc[v, "XS-deseason"], 2, sign=True),
                ]
                for v in rrow.index
            ],
            columns=["Variant (seasonally adjusted signal)", "TS Sharpe", "XS Sharpe"],
        )
    )

    def fig(path: Path) -> None:
        fg, (ax1, ax2) = _figure(1, 2, (11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
        for color, key in zip(SLOTS, labels, strict=True):
            nav = navs[key]
            ax1.plot(nav.index, nav.to_numpy(), color=color, linewidth=1.3, label=labels[key])
        ax1.set_yscale("log")
        ticks = [0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5]
        ax1.yaxis.set_major_locator(FixedLocator(ticks))
        ax1.yaxis.set_minor_locator(NullLocator())
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax1.axhline(1.0, color=AXIS, linewidth=0.8)
        ax1.set_title("Growth of 1 (log scale), no costs", fontsize=10, loc="left")
        ax1.legend(frameon=False, fontsize=8, ncols=2, loc="upper left")
        for color, key in zip(SLOTS, labels, strict=True):
            s = costs[costs["strategy"] == key]
            ax2.plot(
                s["cost_bps"],
                s["sharpe"],
                color=color,
                linewidth=1.5,
                marker="o",
                markersize=4,
                label=labels[key],
            )
        ax2.axhline(0.0, color=AXIS, linewidth=0.8)
        ax2.set_xlabel("cost, bp of traded notional")
        ax2.set_title("Sharpe ratio vs trading cost", fontsize=10, loc="left")
        fg.tight_layout()
        _save(fg, path, "Carry strategies on four energy futures")

    out.figures["m2_carry.png"] = fig
    return {
        "start": start,
        "xs_sharpe": float(metrics.set_index("strategy").loc["XS-deseason", "sharpe"]),
        "xs_t": float(metrics.set_index("strategy").loc["XS-deseason", "t_stat"]),
        "ts_sharpe": float(metrics.set_index("strategy").loc["TS-deseason", "sharpe"]),
        "ts_t": float(metrics.set_index("strategy").loc["TS-deseason", "t_stat"]),
        "lo_sharpe": float(metrics.set_index("strategy").loc["LO", "sharpe"]),
        "years": float(years),
        "first": first,
        "last": last,
    }


# ---------------------------------------------------------------- module 3


def _storage_figure(
    path: Path,
    *,
    contract: StorageContract,
    strip: pd.DataFrame,
    curves_: tuple[np.ndarray, np.ndarray],
    plan: IntrinsicPlan,
    valuation: tuple[StorageValuation, RollingResult],
    t0: pd.Timestamp,
) -> None:
    """Market vs seasonal curve, intrinsic schedule, distribution of RI path values."""
    market, seasonal = curves_
    base_val, base_res = valuation
    fg, (ax1, ax2, ax3) = _figure(1, 3, (13, 3.9), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    months = contract.months.to_timestamp()
    all_months = pd.to_datetime(strip["delivery_month"])
    ax1.plot(
        all_months,
        strip["settle"],
        color=AXIS,
        linewidth=1.0,
        marker="o",
        markersize=3,
        label="NYMEX strip (all listed)",
    )
    ax1.plot(
        months,
        market,
        color=SLOTS[0],
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="contract months",
    )
    ax1.plot(
        months,
        seasonal,
        color=SLOTS[1],
        linewidth=1.5,
        marker="s",
        markersize=3.5,
        label="average seasonal shape",
    )
    ax1.xaxis.set_major_locator(MonthLocator(bymonth=(1, 7)))
    ax1.xaxis.set_major_formatter(DateFormatter("%b\n%Y"))
    ax1.set_ylabel("USD/MMBtu")
    ax1.set_title(f"Henry Hub curve on {t0.date()}", fontsize=10, loc="left")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    x = np.arange(contract.n_months)
    ax2.bar(x, plan.injection, color=SLOTS[0], width=0.6, label="injection")
    ax2.bar(x, -plan.withdrawal, color=SLOTS[1], width=0.6, label="withdrawal")
    ax2.step(
        x,
        plan.inventory,
        where="mid",
        color=INK_2,
        linewidth=1.3,
        label="inventory (end of month)",
    )
    ax2.axhline(0, color=AXIS, linewidth=0.8)
    ax2.set_xticks(x, [m.strftime("%b\n%y") for m in months], fontsize=8)
    ax2.set_ylabel("share of capacity")
    ax2.set_title("Intrinsic schedule on the market strip", fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    ax3.hist(base_res.value, bins=50, color=SLOTS[0], alpha=0.85)
    ax3.axvline(base_val.intrinsic, color=INK, linewidth=1.2)
    ax3.annotate(
        "intrinsic",
        (base_val.intrinsic, ax3.get_ylim()[1] * 0.9),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
        color=INK,
    )
    ax3.set_xlabel("rolling-intrinsic value per path, USD/MMBtu")
    ax3.set_title("Rolling intrinsic, base model", fontsize=10, loc="left")
    fg.tight_layout()
    _save(fg, path, "Natural gas storage valuation")


def module3(panel: pd.DataFrame, out: Output) -> dict[str, float]:
    contract = StorageContract()
    strip = pd.read_parquet(SAMPLE / "ng_strip_snapshot.parquet")
    rates = pd.read_parquet(SAMPLE / "rates_daily.parquet")
    t0 = pd.Timestamp(NG_STRIP_SNAPSHOT_DATE)
    dtb3 = rates[(rates["series_id"] == "DTB3") & (rates["date"] <= t0)].iloc[-1]
    rate = tbill_continuous_rate(float(dtb3["value"]))
    market = strip_forwards(strip, contract)
    spreads = seasonal_log_spreads(panel)
    seasonal = chained_seasonal_curve(spreads, contract, level=float(market.mean()))
    disc = discount_factors(contract, t0, rate)
    plan = intrinsic_lp(contract, market, disc)
    plan_seasonal = intrinsic_lp(contract, seasonal, disc)

    calib = {}
    for label, window in {
        "2014-04 to 2024-04 (base)": ("2014-04-06", "2024-04-06"),
        "2019-04 to 2024-04": ("2019-04-06", "2024-04-06"),
    }.items():
        calib[label] = calibrate(calibration_sample(panel, "NG", *window))
    base_label = "2014-04 to 2024-04 (base)"
    base = calib[base_label].params
    alt = calib["2019-04 to 2024-04"].params
    variants: dict[str, TwoFactorParams] = {
        "base (calibrated 2014–2024)": base,
        "calibrated 2019–2024": alt,
        "κ × 0.5": base.scaled(kappa=0.5),
        "κ × 2": base.scaled(kappa=2.0),
        "σ_s × 0.5": base.scaled(sigma_s=0.5),
        "σ_s × 1.5": base.scaled(sigma_s=1.5),
        "σ_l × 0.5": base.scaled(sigma_l=0.5),
        "σ_l × 1.5": base.scaled(sigma_l=1.5),
        "ρ = 0": base.scaled(rho=0.0),
        "ρ = +0.5": base.scaled(rho=0.5),
    }
    sens = sensitivity(contract, market, t0, rate, variants, n_paths=STORAGE_PATHS, seed=SEED)
    seasonal_val, _ = value_storage(
        contract, seasonal, t0, rate, base, n_paths=STORAGE_PATHS, rng=np.random.default_rng(SEED)
    )
    base_val, base_res = value_storage(
        contract, market, t0, rate, base, n_paths=STORAGE_PATHS, rng=np.random.default_rng(SEED)
    )
    sens.insert(0, "curve", "market strip")
    seasonal_row = {
        "curve": "average seasonal shape",
        "variant": "base (calibrated 2014–2024)",
        "sigma_s": base.sigma_s,
        "kappa": base.kappa,
        "sigma_l": base.sigma_l,
        "rho": base.rho,
        "intrinsic": seasonal_val.intrinsic,
        "rolling_intrinsic": seasonal_val.rolling_mean,
        "se": seasonal_val.rolling_se,
        "extrinsic": seasonal_val.extrinsic,
        "extrinsic_share": seasonal_val.extrinsic_share,
    }
    sens = pd.concat([sens, pd.DataFrame([seasonal_row])], ignore_index=True)
    out.tables["storage_valuation"] = sens
    out.tables["storage_plan"] = pd.DataFrame(
        {
            "delivery_month": contract.months.astype(str),
            "market_forward": market,
            "seasonal_forward": seasonal,
            "injection": plan.injection,
            "withdrawal": plan.withdrawal,
            "inventory_end": plan.inventory,
            "discount": disc,
        }
    )
    cal_rows = []
    for label, res in calib.items():
        p = res.params
        cal_rows.append(
            {
                "window": label,
                "sigma_s": p.sigma_s,
                "kappa": p.kappa,
                "sigma_l": p.sigma_l,
                "rho": p.rho,
                "rmse_corr_scale": res.rmse_corr_scale,
                "n_days": res.n_days,
                "emp_vol_c1": float(np.sqrt(res.empirical[0, 0])),
                "fit_vol_c1": float(np.sqrt(res.fitted[0, 0])),
                "emp_vol_c4": float(np.sqrt(res.empirical[3, 3])),
                "fit_vol_c4": float(np.sqrt(res.fitted[3, 3])),
                "emp_corr_c1c2": float(
                    res.empirical[0, 1] / np.sqrt(res.empirical[0, 0] * res.empirical[1, 1])
                ),
                "fit_corr_c1c2": float(
                    res.fitted[0, 1] / np.sqrt(res.fitted[0, 0] * res.fitted[1, 1])
                ),
            }
        )
    cal = pd.DataFrame(cal_rows)
    out.tables["storage_calibration"] = cal
    out.tables["storage_seasonal_spreads"] = spreads

    out.blocks["m3-calibration"] = md_table(
        pd.DataFrame(
            [
                [
                    r.window,
                    num(r.sigma_s, 3),
                    num(r.kappa, 2),
                    num(r.sigma_l, 3),
                    num(r.rho, 2, sign=True),
                    f"{num(r.emp_vol_c1, 2)} / {num(r.fit_vol_c1, 2)}",
                    f"{num(r.emp_vol_c4, 2)} / {num(r.fit_vol_c4, 2)}",
                    f"{num(r.emp_corr_c1c2, 3)} / {num(r.fit_corr_c1c2, 3)}",
                    f"{r.n_days}",
                ]
                for r in cal.itertuples()
            ],
            columns=[
                "NG window",
                "σ_s",
                "κ",
                "σ_l",
                "ρ",
                "vol C1 data / model",
                "vol C4 data / model",
                "corr C1–C2 data / model",
                "days",
            ],
        )
    )
    srows = []
    for r in sens.itertuples():
        srows.append(
            [
                r.curve,
                r.variant,
                num(r.intrinsic, 3),
                f"{num(r.rolling_intrinsic, 3)} ± {num(r.se, 3)}",
                num(r.extrinsic, 3),
                pct(r.extrinsic_share, 1, sign=True),
            ]
        )
    out.blocks["m3-table"] = (
        md_table(
            pd.DataFrame(
                srows,
                columns=[
                    "Curve",
                    "Model parameters",
                    "Intrinsic",
                    "Rolling intrinsic ± s.e.",
                    "Extrinsic",
                    "Extrinsic / intrinsic",
                ],
            ),
            align="llrrrr",
        )
        + f"\n\nUSD/MMBtu of working-gas capacity. Contract months {contract.months[0]} to "
        f"{contract.months[-1]}, valued on {t0.date()} (NG strip snapshot), discount rate "
        f"{pct(rate, 2)} (DTB3 of {dtb3['date'].date()}, continuous). "
        f"{STORAGE_PATHS:,} antithetic paths, {base_val.n_decisions} decision dates, "
        "common random numbers across rows."
    )

    out.figures["m3_storage.png"] = partial(
        _storage_figure,
        contract=contract,
        strip=strip,
        curves_=(market, seasonal),
        plan=plan,
        valuation=(base_val, base_res),
        t0=t0,
    )
    return {
        "intrinsic": base_val.intrinsic,
        "ri": base_val.rolling_mean,
        "ri_se": base_val.rolling_se,
        "share": base_val.extrinsic_share,
        "seasonal_intrinsic": plan_seasonal.value,
        "start": str(contract.months[0]),
        "end": str(contract.months[-1]),
    }


# ---------------------------------------------------------------- module 4


def module4(panel: pd.DataFrame, out: Output) -> dict[str, float]:
    legs = [calibrate_leg(panel, code) for code in LEGS]
    corr = leg_correlations(panel, LEGS)
    rates = pd.read_parquet(SAMPLE / "rates_daily.parquet")
    t0 = pd.Timestamp(SPREAD_VALUATION)
    dtb3 = rates[(rates["series_id"] == "DTB3") & (rates["date"] <= t0)].iloc[-1]
    rate = tbill_continuous_rate(float(dtb3["value"]))
    snaps = [snapshot(panel, SPREAD_VALUATION, m) for m in SPREAD_MONTHS]
    grid = price_grid(
        snaps,
        legs,
        corr.to_numpy(),
        rate,
        SPREAD_OFFSETS,
        n_paths=SPREAD_PATHS,
        rng=np.random.default_rng(SEED),
    )
    out.tables["spread_options"] = grid
    params = pd.DataFrame(
        {
            "leg": LEGS,
            "sigma": [p.sigma for p in legs],
            "kappa": [p.kappa for p in legs],
        }
    )
    out.tables["spread_legs"] = params
    out.tables["spread_correlations"] = corr.reset_index(names="leg")

    leg_rows = [
        [code, num(p.sigma, 3), num(p.kappa, 2)]
        + [num(float(corr.loc[code, other]), 3) for other in LEGS]
        for code, p in zip(LEGS, legs, strict=True)
    ]
    out.blocks["m4-legs"] = md_table(
        pd.DataFrame(leg_rows, columns=["Leg", "σ", "κ", *(f"ρ with {c}" for c in LEGS)])
    )
    rows = []
    for (spread, month), g in grid.groupby(["spread", "delivery_month"], sort=False):
        atm = g[g["moneyness"] == 0.0].iloc[0]
        worst = g.loc[g["error_pct"].abs().idxmax()]
        rows.append(
            [
                spread,
                month,
                num(atm["option_expiry_years"] * 12, 1),
                num(atm["forward_spread"], 2),
                f"{num(atm['mc_price'], 3)} ± {num(atm['mc_se'], 3)}",
                num(atm["kirk_price"], 3),
                num(atm["error_in_se"], 1, sign=True),
                f"{pct(worst['error_pct'] / 100, 2, sign=True)} at K = F{worst['moneyness']:+g}",
                num(g["error_in_se"].abs().max(), 1),
            ]
        )
    out.blocks["m4-table"] = (
        md_table(
            pd.DataFrame(
                rows,
                columns=[
                    "Spread (USD/bbl)",
                    "Delivery",
                    "Expiry (months)",
                    "Forward spread",
                    "ATM call, MC ± s.e.",
                    "ATM call, Kirk",
                    "ATM error (s.e.)",
                    "Largest error in %",
                    "Largest |error| (s.e.)",
                ],
            ),
            align="llrrrrrrr",
        )
        + f"\n\nValued on {t0.date()}; strikes F{SPREAD_OFFSETS[0]:+g} to "
        f"F{SPREAD_OFFSETS[-1]:+g} USD/bbl around the forward spread F; {SPREAD_PATHS:,} "
        f"antithetic draws per delivery month; discount rate {pct(rate, 2)}. "
        "Full grid in `docs/results/spread_options.csv`."
    )

    def fig(path: Path) -> None:
        fg, axes = _figure(1, 2, (11, 3.8), sharey=True)
        for ax, (spread, g) in zip(axes, grid.groupby("spread", sort=False), strict=True):
            ax.axhspan(-2, 2, color=GRID, alpha=0.6, linewidth=0)
            ax.axhline(0, color=AXIS, linewidth=0.8)
            for color, (month, h) in zip(SLOTS, g.groupby("delivery_month"), strict=False):
                ax.plot(
                    h["moneyness"],
                    h["error_in_se"],
                    color=color,
                    linewidth=1.5,
                    marker="o",
                    markersize=4,
                    label=f"{month} delivery",
                )
            ax.set_title(spread, fontsize=10, loc="left")
            ax.set_xlabel("strike − forward spread, USD/bbl")
        axes[0].set_ylabel("(Kirk − MC) / MC standard error")
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        fg.suptitle(
            "Kirk approximation error; gray band = ±2 Monte Carlo standard errors",
            fontsize=11,
            color=INK,
            x=0.01,
            ha="left",
        )
        fg.tight_layout()
        _save(fg, path, "Kirk versus Monte Carlo on crack spreads")

    out.figures["m4_kirk_vs_mc.png"] = fig
    worst = grid.loc[grid["error_pct"].abs().idxmax()]
    atm = grid[grid["moneyness"] == 0.0]
    return {
        "worst_pct": float(worst["error_pct"]),
        "worst_spread": str(worst["spread"]),
        "worst_offset": float(worst["moneyness"]),
        "worst_month": str(worst["delivery_month"]),
        "atm_max_abs_pct": float(atm["error_pct"].abs().max()),
    }


# ---------------------------------------------------------------- SQL layer


def sql_layer(out: Output) -> dict[str, float]:
    con = build_database()
    inv = run_query(con, "33_inventories", start="2010-01-01", end=EIA_END)
    summary = (
        inv.groupby("code")
        .agg(
            region=("region", "first"),
            n_weeks=("n_weeks", "first"),
            corr_y14_dev=("corr_y14_dev", "first"),
            slope_y14_dev=("slope_y14_dev", "first"),
        )
        .reindex(list(CODES_ORDER))
    )
    con.close()
    out.tables["sql_inventory_vs_curve"] = summary.reset_index()
    out.blocks["sql-example"] = (
        md_table(
            pd.DataFrame(
                [
                    [
                        code,
                        r.region,
                        f"{int(r.n_weeks)}",
                        num(r.corr_y14_dev, 2, sign=True),
                        num(100 * r.slope_y14_dev, 2, sign=True),
                    ]
                    for code, r in summary.iterrows()
                ],
                columns=[
                    "Contract",
                    "Stocks",
                    "Weeks",
                    "corr(y14, stock gap)",
                    "y14 change per +1 pt of stock gap (pp/yr)",
                ],
            )
        )
        + "\n\nWeekly EIA stocks versus the average of the same week over the 5 previous "
        "years, 2010 to 2024-04; y14 = annualised slope between contracts 1 and 4 on the last "
        "trading day of the week."
    )
    return {
        "corr_min": float(summary["corr_y14_dev"].min()),
        "corr_max": float(summary["corr_y14_dev"].max()),
    }


# ---------------------------------------------------------------- summary + data


def key_results(
    out: Output, term: pd.DataFrame, *, m2: dict, m3: dict, m4: dict, sql: dict
) -> None:
    full = term[term["period"] == period_label(*M1_PERIODS[0])].set_index("code")
    ry_cells = " · ".join(f"{c} {pct(full.loc[c, 'mean_y12'], 1, sign=True)}" for c in CODES_ORDER)
    bw_cells = " · ".join(f"{c} {pct(full.loc[c, 'share_backwardation'], 0)}" for c in CODES_ORDER)
    rows = [
        ["Term structure", f"Mean front roll yield y12, {period_label(*M1_PERIODS[0])}", ry_cells],
        ["Term structure", "Share of days in backwardation", bw_cells],
        [
            "Inventories (SQL)",
            "Correlation of curve slope y14 with the 5-year stock gap, 2010–2024",
            f"{num(sql['corr_min'], 2, sign=True)} to {num(sql['corr_max'], 2, sign=True)}",
        ],
        [
            "Carry",
            f"Sharpe, seasonally adjusted signal, {m2['first'][:4]}–{m2['last'][:4]} (t-stat)",
            f"XS {num(m2['xs_sharpe'], 2, sign=True)} (t {num(m2['xs_t'], 2, sign=True)}) · "
            f"TS {num(m2['ts_sharpe'], 2, sign=True)} (t {num(m2['ts_t'], 2, sign=True)}) · "
            f"long only {num(m2['lo_sharpe'], 2, sign=True)}",
        ],
        [
            "Gas storage",
            f"Intrinsic vs rolling intrinsic, {m3['start']} to {m3['end']} (USD/MMBtu of capacity)",
            f"{num(m3['intrinsic'], 3)} vs {num(m3['ri'], 3)} ± {num(m3['ri_se'], 3)} "
            f"(extrinsic {pct(m3['share'], 1, sign=True)})",
        ],
        [
            "Crack-spread options",
            "Kirk − Monte Carlo: largest ATM error · largest error on the grid",
            f"{pct(m4['atm_max_abs_pct'] / 100, 2)} · {pct(m4['worst_pct'] / 100, 2, sign=True)} "
            f"({m4['worst_spread']}, {m4['worst_month']}, K = F{m4['worst_offset']:+g})",
        ],
    ]
    out.blocks["key-results"] = md_table(
        pd.DataFrame(rows, columns=["Module", "Quantity", "Result"]), align="lll"
    )


def data_coverage(out: Output) -> None:
    manifest = json.loads((SAMPLE / "MANIFEST.json").read_text())
    rows = []
    for name, entry in manifest["files"].items():
        extracted = sorted({s["extracted_at_utc"][:10] for s in entry["sources"]})
        rows.append(
            [
                f"`{name}`",
                f"{entry['rows']:,}",
                entry["first_date"],
                entry["last_date"],
                ", ".join(extracted),
            ]
        )
    out.blocks["data-coverage"] = md_table(
        pd.DataFrame(rows, columns=["File", "Rows", "First", "Last", "Extracted (UTC)"]),
        align="lrlll",
    )


# ---------------------------------------------------------------- documents


def render(text: str, blocks: dict[str, str]) -> tuple[str, set[str]]:
    """Replace each ``<!-- BEGIN:name -->...<!-- END:name -->`` block; return used names."""
    used: set[str] = set()

    def sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in blocks:
            raise KeyError(f"no generated content for block {name!r}")
        used.add(name)
        return f"<!-- BEGIN:{name} -->\n{blocks[name]}\n<!-- END:{name} -->"

    pattern = re.compile(r"<!-- BEGIN:([\w-]+) -->.*?<!-- END:\1 -->", re.DOTALL)
    return pattern.sub(sub, text), used


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if a document block is stale")
    args = parser.parse_args()
    t_start = time.time()
    out = Output()
    panel = build_curve_panel(pd.read_parquet(SAMPLE / "futures_daily.parquet"))
    ry = module1(panel, out)
    m2 = module2(panel, ry, out)
    m3 = module3(panel, out)
    m4 = module4(panel, out)
    sql = sql_layer(out)
    key_results(out, out.tables["term_structure"], m2=m2, m3=m3, m4=m4, sql=sql)
    data_coverage(out)

    used: set[str] = set()
    stale = []
    for doc in DOCUMENTS:
        old = doc.read_text()
        new, names = render(old, out.blocks)
        used |= names
        if new != old:
            stale.append(doc.name)
            if not args.check:
                doc.write_text(new)
    unused = set(out.blocks) - used
    if unused:
        print(f"warning: blocks not placed in any document: {sorted(unused)}")
    if args.check:
        print(f"stale documents: {stale}" if stale else "documents are up to date")
        return 1 if stale else 0

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, table in out.tables.items():
        table.to_csv(RESULTS / f"{name}.csv", index=False, float_format="%.10g")
    for name, writer in out.figures.items():
        writer(FIGURES / name)
    print(f"updated: {stale or 'nothing'}; {len(out.tables)} tables, {len(out.figures)} figures")
    print(f"done in {time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

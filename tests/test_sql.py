"""SQL layer: idempotent build, and Python/SQL implementations agree exactly."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from energycurves import curves
from energycurves.data.stitch import slot_returns
from energycurves.db import build_database, run_query, table_counts

FULL = {"start": "1980-01-01", "end": "2024-04-06"}


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    return build_database()


def _snapshot(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    out = {}
    for table in table_counts(con)["table"]:
        frame = con.execute(f'SELECT * FROM "{table}"').df()
        out[table] = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    return out


def test_two_builds_give_the_same_database(tmp_path: Path) -> None:
    path = tmp_path / "twice.duckdb"
    first = build_database(path)
    before = _snapshot(first)
    first.close()
    second = build_database(path)  # same file: upserts + full refresh
    after = _snapshot(second)
    second.close()
    assert before.keys() == after.keys()
    for table, frame in before.items():
        pd.testing.assert_frame_equal(frame, after[table], check_exact=True, obj=table)


def test_rebuild_removes_rows_that_left_the_sample(tmp_path: Path) -> None:
    path = tmp_path / "stale.duckdb"
    first = build_database(path)
    fresh = table_counts(first)
    first.execute("INSERT INTO dim_business_day VALUES (DATE '1970-01-02')")
    first.execute(
        "INSERT INTO fact_price_daily (date, code, contract, settle, volume, source) "
        "VALUES (DATE '1970-01-02', 'CL', 1, 1.0, NULL, 'stale')"
    )
    first.close()
    second = build_database(path)  # upserts alone would keep both stale rows
    stale = second.execute(
        "SELECT (SELECT count(*) FROM dim_business_day WHERE date = DATE '1970-01-02') "
        "+ (SELECT count(*) FROM fact_price_daily WHERE source = 'stale')"
    ).fetchone()
    rebuilt = table_counts(second)
    second.close()
    assert stale == (0,)
    pd.testing.assert_frame_equal(fresh, rebuilt)


def test_fact_curve_equals_python_panel(
    con: duckdb.DuckDBPyConnection, panel: pd.DataFrame
) -> None:
    sql = con.execute(
        "SELECT trade_date AS date, code, contract, delivery_month, last_trade_date, tau, settle "
        "FROM fact_curve ORDER BY code, date, contract"
    ).df()
    py = panel.sort_values(["code", "date", "contract"]).reset_index(drop=True)
    assert len(sql) == len(py)
    for col in ("date", "delivery_month", "last_trade_date"):
        assert (pd.to_datetime(sql[col]).to_numpy() == py[col].to_numpy()).all(), col
    assert (sql["code"].to_numpy() == py["code"].to_numpy()).all()
    assert (sql["contract"].to_numpy() == py["contract"].to_numpy()).all()
    np.testing.assert_allclose(sql["tau"], py["tau"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(sql["settle"], py["settle"])


def test_slot_returns_view_equals_python(
    con: duckdb.DuckDBPyConnection, panel: pd.DataFrame
) -> None:
    sql = con.execute(
        "SELECT trade_date AS date, code, ret, held_month, rolled FROM v_slot_returns "
        "WHERE slot = 2"
    ).df()
    py = slot_returns(panel, slot=2)
    merged = py.merge(sql, on=["date", "code"], suffixes=("_py", "_sql"), validate="one_to_one")
    assert len(merged) == len(py) == len(sql)  # every trading day, missing returns included
    np.testing.assert_allclose(merged["ret_py"], merged["ret_sql"], rtol=1e-12, atol=1e-14)
    held_sql = pd.to_datetime(merged["held_month_sql"]).to_numpy()
    assert (merged["held_month_py"].to_numpy() == held_sql).all()
    assert (merged["rolled_py"] == merged["rolled_sql"]).all()


def test_roll_yield_view_equals_python(con: duckdb.DuckDBPyConnection, panel: pd.DataFrame) -> None:
    sql = con.execute("SELECT trade_date AS date, code, y12, y23, y14 FROM v_roll_yield").df()
    py = curves.roll_yields(panel)
    merged = py.merge(sql, on=["date", "code"], suffixes=("_py", "_sql"), validate="one_to_one")
    assert len(merged) == len(py)
    for col in ("y12", "y23", "y14"):
        a, b = merged[f"{col}_py"].to_numpy(float), merged[f"{col}_sql"].to_numpy(float)
        assert (np.isnan(a) == np.isnan(b)).all(), col
        np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-12)


def test_term_structure_query_equals_python(
    con: duckdb.DuckDBPyConnection, panel: pd.DataFrame
) -> None:
    params = {"start": "2008-01-01", "end": "2024-04-06", "band": 0.01}
    sql = run_query(con, "30_term_structure", **params).set_index("code")
    py = curves.term_structure_summary(
        curves.roll_yields(panel), params["start"], params["end"], params["band"]
    ).set_index("code")
    cols = ["n_days", "mean_y12", "median_y12", "share_backwardation", "share_contango", "mean_y14"]
    pd.testing.assert_frame_equal(
        sql[cols].astype(float), py[cols].astype(float), rtol=1e-10, check_names=False
    )


def test_seasonality_query_equals_python(
    con: duckdb.DuckDBPyConnection, panel: pd.DataFrame
) -> None:
    sql = run_query(con, "32_seasonality", start="2008-01-01", end="2024-04-06")
    py = curves.seasonal_profile(curves.roll_yields(panel), "2008-01-01", "2024-04-06")
    merged = py.merge(sql, on=["code", "front_month"], suffixes=("_py", "_sql"))
    assert len(merged) == 48
    np.testing.assert_allclose(merged["mean_y12_py"], merged["mean_y12_sql"], rtol=1e-10)
    assert (merged["n_days_py"] == merged["n_days_sql"]).all()


def test_crack_spread_legs_are_matched_on_delivery_month(con: duckdb.DuckDBPyConnection) -> None:
    out = run_query(con, "31_spread_zscore", start="2023-06-19", end="2023-06-24")
    rows = out[out["spread"] == "crack_321"].set_index("trade_date")
    # 2023-06-21: the CL July contract expired the day before, RB/HO July still trade,
    # so the shown month (September) is CL contract 2 but RB/HO contract 3.
    day = pd.Timestamp("2023-06-21")
    month = rows.loc[day, "delivery_month"]
    legs = con.execute(
        "SELECT code, contract, settle FROM fact_curve WHERE trade_date = ? AND delivery_month = ?",
        [day, month],
    ).df()
    slots = legs.set_index("code")["contract"]
    prices = legs.set_index("code")["settle"]
    assert (slots["CL"], slots["RB"], slots["HO"]) == (2, 3, 3)
    expected = (2 * 42 * prices["RB"] + 42 * prices["HO"]) / 3 - prices["CL"]
    assert np.isclose(rows.loc[day, "value"], expected)


def _gasoline_crack_level(con: duckdb.DuckDBPyConnection, shown: pd.DataFrame) -> np.ndarray:
    """Back-adjusted level rebuilt in pandas: cumulated changes of yesterday's month."""
    curve = con.execute(
        "SELECT trade_date, code, delivery_month, settle FROM fact_curve WHERE code IN ('CL', 'RB')"
    ).df()
    px = curve.pivot_table(index=["trade_date", "delivery_month"], columns="code", values="settle")
    crack = (42 * px["RB"] - px["CL"]).dropna()
    prev_day = shown["trade_date"].shift(1)
    prev_month = shown["delivery_month"].shift(1)
    now = crack.reindex(pd.MultiIndex.from_arrays([shown["trade_date"], prev_month])).to_numpy()
    before = crack.reindex(pd.MultiIndex.from_arrays([prev_day, prev_month])).to_numpy()
    return np.cumsum(np.nan_to_num(now - before))


def test_spread_zscore_exec_window_ends_yesterday(con: duckdb.DuckDBPyConnection) -> None:
    out = run_query(con, "31_spread_zscore", **FULL)
    s = out[out["spread"] == "gasoline_crack"].reset_index(drop=True)
    level = _gasoline_crack_level(con, s)
    for i in (500, 2500):
        past = level[i - 60 : i]
        expected = (level[i] - past.mean()) / past.std(ddof=1)
        assert np.isclose(s["z60_exec"].iloc[i], expected)
    assert s["z60_exec"].iloc[:60].isna().all()


def test_spread_zscore_ignores_the_change_of_delivery_month(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """At each CL expiry the shown month changes; the z-score must not jump with it."""
    out = run_query(con, "31_spread_zscore", **FULL)
    for spread, grp in out.groupby("spread"):
        s = grp.sort_values("trade_date")
        dz = s["z60"].diff().abs()
        switch = s["delivery_month"].ne(s["delivery_month"].shift()) & dz.notna()
        assert switch.sum() > 200
        assert dz[switch].mean() < 1.5 * dz[~switch].mean(), spread


def test_inventory_deviation_uses_past_years_only(con: duckdb.DuckDBPyConnection) -> None:
    out = run_query(con, "33_inventories", start="2015-01-01", end="2024-04-06")
    inv = con.execute("SELECT week_ending, stock FROM fact_inventory WHERE code = 'CL'").df()
    inv["week_ending"] = pd.to_datetime(inv["week_ending"])
    inv["year"] = inv["week_ending"].dt.isocalendar().year
    inv["week"] = inv["week_ending"].dt.isocalendar().week
    row = out[(out["code"] == "CL")].iloc[100]
    when = pd.Timestamp(row["week_ending"])
    year, week = when.isocalendar().year, when.isocalendar().week
    hist = inv[(inv["week"] == week) & (inv["year"] >= year - 5) & (inv["year"] <= year - 1)]
    assert len(hist) == 5
    expected = 100 * (row["stock"] / hist["stock"].mean() - 1)
    assert np.isclose(row["dev_5y_pct"], expected)
    assert pd.Timestamp(row["curve_date"]) <= when


def test_fx_asof_join_keeps_every_trading_day(con: duckdb.DuckDBPyConnection) -> None:
    out = run_query(con, "34_fx_conversion", start="2010-01-01", end="2024-04-06")
    n_days = con.execute(
        "SELECT count(*) FROM fact_curve WHERE code = 'CL' AND contract = 2 "
        "AND trade_date >= DATE '2010-01-01' AND trade_date < DATE '2024-04-06'"
    ).fetchone()
    assert n_days is not None
    assert out["n_days"].sum() == n_days[0]
    assert (out["wti_eur_per_bbl"] < out["wti_usd_per_bbl"]).mean() > 0.9  # EUR > USD mostly

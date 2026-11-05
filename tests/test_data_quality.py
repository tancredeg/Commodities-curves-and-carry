"""Data-quality controls, written in SQL and run against the rebuilt database."""

from __future__ import annotations

import duckdb
import pytest

from energycurves.db import build_database

PRIMARY_KEYS = {
    "dim_commodity": ["code"],
    "dim_contract_calendar": ["code", "delivery_month"],
    "fact_price_daily": ["date", "code", "contract"],
    "fact_curve": ["trade_date", "code", "contract"],
    "fact_curve_snapshot": ["snapshot_date", "code", "delivery_month"],
    "fact_spot_daily": ["date", "series_id"],
    "fact_inventory": ["week_ending", "code", "region"],
    "fact_rate_daily": ["date", "series_id"],
    "fx_rates": ["date", "pair"],
}


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    return build_database()


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> object:
    row = con.execute(sql).fetchone()
    assert row is not None
    return row[0]


@pytest.mark.parametrize(("table", "key"), PRIMARY_KEYS.items())
def test_keys_are_unique(con: duckdb.DuckDBPyConnection, table: str, key: list[str]) -> None:
    cols = ", ".join(key)
    dupes = scalar(
        con,
        f"SELECT count(*) FROM (SELECT {cols} FROM {table} GROUP BY {cols} HAVING count(*) > 1)",
    )
    assert dupes == 0


def test_mandatory_columns_have_no_nulls(con: duckdb.DuckDBPyConnection) -> None:
    checks = {
        "fact_curve": "settle IS NULL OR delivery_month IS NULL OR tau IS NULL OR source IS NULL",
        "fact_inventory": "stock IS NULL OR unit IS NULL",
        "fact_spot_daily": "price IS NULL OR unit IS NULL",
    }
    for table, cond in checks.items():
        assert scalar(con, f"SELECT count(*) FROM {table} WHERE {cond}") == 0, table


def test_every_code_exists_in_dim_commodity(con: duckdb.DuckDBPyConnection) -> None:
    for table in ("fact_price_daily", "fact_curve", "fact_inventory", "fact_curve_snapshot"):
        orphans = scalar(
            con,
            f"SELECT count(*) FROM {table} AS f ANTI JOIN dim_commodity AS d ON d.code = f.code",
        )
        assert orphans == 0, table


def test_no_gap_longer_than_two_business_days(con: duckdb.DuckDBPyConnection) -> None:
    worst = scalar(
        con,
        """
        WITH days AS (
            SELECT code, trade_date,
                   lag(trade_date) OVER (PARTITION BY code ORDER BY trade_date) AS prev
            FROM (SELECT DISTINCT code, trade_date FROM fact_curve)
        )
        SELECT max(missing) FROM (
            SELECT count(b.date) AS missing
            FROM days AS d
            LEFT JOIN dim_business_day AS b ON b.date > d.prev AND b.date < d.trade_date
            WHERE d.prev IS NOT NULL
            GROUP BY d.code, d.trade_date
        )
        """,
    )
    assert worst <= 2


def test_only_known_negative_price(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT code, trade_date, contract FROM fact_curve WHERE settle <= 0"
    ).fetchall()
    assert [(c, str(d), k) for c, d, k in rows] == [("CL", "2020-04-20", 1)]


def test_extreme_returns_are_reviewed_market_events(con: duckdb.DuckDBPyConnection) -> None:
    """Same-contract daily moves beyond ±25 %: each one checked against history."""
    rows = con.execute(
        "SELECT code, CAST(trade_date AS VARCHAR) FROM v_daily_returns WHERE abs(ret) > 0.25"
    ).fetchall()
    reviewed = {
        ("CL", "2020-04-20"),  # May 2020 WTI settles at -37.63 on its eve of expiry
        ("CL", "2020-04-21"),
        ("CL", "2020-04-30"),
        ("RB", "2020-03-23"),  # Covid demand collapse
        ("RB", "2020-04-22"),
        ("NG", "2022-01-27"),  # February 2022 contract squeezed at expiry
        ("NG", "2003-02-24"),  # February 2003 cold snap
        ("CL", "1991-01-17"),  # Gulf War air campaign: oil collapses
        ("HO", "1991-01-17"),
        ("HO", "1990-09-04"),  # Gulf War run-up (sparse early HO data: multi-day moves)
        ("HO", "1990-10-01"),
        ("HO", "1990-01-02"),  # winter 1989-90 heating-oil spike unwinding
        ("HO", "1986-02-03"),  # 1986 oil price collapse
        ("HO", "1986-09-02"),  # OPEC output agreement, August 1986
    }
    assert set(rows) <= reviewed


def test_one_unit_per_commodity_and_series(con: duckdb.DuckDBPyConnection) -> None:
    assert scalar(con, "SELECT count(DISTINCT unit) FROM dim_commodity WHERE code = 'CL'") == 1
    multi = scalar(
        con,
        "SELECT count(*) FROM (SELECT series_id FROM fact_spot_daily "
        "GROUP BY series_id HAVING count(DISTINCT unit) > 1)",
    )
    assert multi == 0
    multi_inv = scalar(
        con,
        "SELECT count(*) FROM (SELECT code, region FROM fact_inventory "
        "GROUP BY code, region HAVING count(DISTINCT unit) > 1)",
    )
    assert multi_inv == 0


def test_curve_build_does_not_multiply_rows(con: duckdb.DuckDBPyConnection) -> None:
    raw_on_bd = scalar(
        con,
        "SELECT count(*) FROM fact_price_daily AS p SEMI JOIN dim_business_day AS b "
        "ON b.date = p.date",
    )
    curve = scalar(con, "SELECT count(*) FROM fact_curve")
    assert isinstance(raw_on_bd, int)
    assert curve == raw_on_bd - 2  # exactly the two isolated typos removed
    slot_rows = scalar(con, "SELECT count(*) FROM v_slot_returns")
    assert isinstance(slot_rows, int) and isinstance(curve, int)
    assert slot_rows <= curve


def test_delivery_months_are_strictly_increasing_along_the_curve(
    con: duckdb.DuckDBPyConnection,
) -> None:
    bad = scalar(
        con,
        """
        SELECT count(*) FROM (
            SELECT
                delivery_month,
                lag(delivery_month) OVER (
                    PARTITION BY code, trade_date ORDER BY contract
                ) AS prev
            FROM fact_curve
        ) WHERE prev IS NOT NULL AND delivery_month <= prev
        """,
    )
    assert bad == 0

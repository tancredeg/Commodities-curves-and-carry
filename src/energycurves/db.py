"""DuckDB layer: build the database from ``sql/`` and run the analytics queries.

Python downloads and traces the data; SQL stores, checks and computes. The
build executes the numbered files of ``sql/`` in order:

``00_schema`` (tables) → ``10_load_*`` (idempotent upserts from
``data/sample``) → ``15_build_curve`` (clean curve, full refresh) →
``20_views`` (returns, curve points, roll yields).

Analytics queries (``sql/analytics/*.sql``) take named parameters
(``$start``, ``$end``...) bound by :func:`run_query`.

Rebuild from the command line::

    python -m energycurves.db            # writes data/commodities.duckdb
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from energycurves.data.calendar import business_days, expiry_calendar
from energycurves.data.registry import CODES, COMMODITIES

REPO_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = REPO_ROOT / "sql"
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
DEFAULT_DB = REPO_ROOT / "data" / "commodities.duckdb"

BUILD_STEPS = ("00_schema", "10_load_*", "15_build_curve", "20_views")


def _reference_frames() -> dict[str, pd.DataFrame]:
    """Reference tables generated in Python and exposed to SQL as ``src_*`` views."""
    commodity = pd.DataFrame([asdict(c) for c in COMMODITIES.values()])
    bdays = pd.DataFrame({"date": business_days("1979-01-01", "2032-12-31")})
    calendar = expiry_calendar(CODES, "1979-01", "2031-12")
    return {
        "src_commodity": commodity,
        "src_business_day": bdays,
        "src_contract_calendar": calendar,
    }


def sql_files() -> list[Path]:
    """Build scripts in execution order."""
    files: list[Path] = []
    for pattern in BUILD_STEPS:
        files.extend(sorted(SQL_DIR.glob(f"{pattern}.sql")))
    return files


def build_database(
    db_path: str | Path = ":memory:", sample_dir: str | Path = SAMPLE_DIR
) -> duckdb.DuckDBPyConnection:
    """Create (or refresh) the database and return an open connection.

    Running it twice on the same file gives the same database: loads are
    upserts on primary keys and derived tables are rebuilt from scratch.
    """
    con = duckdb.connect(str(db_path))
    sample = Path(sample_dir).resolve().as_posix().replace("'", "''")
    con.execute(f"SET VARIABLE sample_dir = '{sample}'")
    references = _reference_frames()
    for name, frame in references.items():
        con.register(name, frame)
    for path in sql_files():
        con.execute(path.read_text())
    for name in references:
        con.unregister(name)
    return con


def run_query(con: duckdb.DuckDBPyConnection, name: str, **params: Any) -> pd.DataFrame:
    """Run ``sql/analytics/<name>.sql`` with named parameters, return a DataFrame."""
    path = SQL_DIR / "analytics" / f"{name}.sql"
    frame: pd.DataFrame = con.execute(path.read_text(), params).df()
    return frame


def table_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Row count of every base table (used by the CLI and the tests)."""
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_type = 'BASE TABLE' ORDER BY table_name"
    ).fetchall()
    rows = [(t, con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]) for (t,) in tables]  # type: ignore[index]
    return pd.DataFrame(rows, columns=["table", "rows"])


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    con = build_database(target)
    print(table_counts(con).to_string(index=False))
    con.close()
    print(f"database written to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

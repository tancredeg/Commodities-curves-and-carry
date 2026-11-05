"""Parsers for the raw public files (EIA ``.xls`` workbooks, FRED ``.csv``).

Every EIA "Download Series History" workbook has the same layout: one sheet
per table (``Data 1``, ``Data 2``...), row 2 holds the source keys, row 3 the
long descriptions, and dates are Excel serial numbers from row 4 onwards.
The parsers return a long DataFrame ``(date, source_key, value)`` so that the
mapping to commodities lives in one place: :mod:`energycurves.data.registry`.
"""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pandas as pd

from energycurves.data.registry import (
    FUTURES_KEYS,
    INVENTORY_SERIES,
    SPOT_SERIES,
    Series,
)

_KEY_ROW = 1
_FIRST_DATA_ROW = 3


def parse_eia_workbook(path: Path) -> pd.DataFrame:
    """Parse every ``Data n`` sheet of an EIA workbook into long format.

    Returns columns ``date`` (datetime64), ``source_key`` (str), ``value`` (float).
    Empty cells are dropped: the EIA leaves a cell blank when no price was
    published for that contract on that day.
    """
    import xlrd  # optional dependency, only needed to rebuild the sample

    book = xlrd.open_workbook(str(path))
    records: list[tuple[dt.datetime, str, float]] = []
    for sheet in book.sheets():
        if not sheet.name.startswith("Data"):
            continue
        keys = [str(sheet.cell_value(_KEY_ROW, c)).strip() for c in range(sheet.ncols)]
        for r in range(_FIRST_DATA_ROW, sheet.nrows):
            raw_date = sheet.cell_value(r, 0)
            if raw_date in ("", None):
                continue
            date = xlrd.xldate_as_datetime(float(raw_date), book.datemode)
            for c in range(1, sheet.ncols):
                cell = sheet.cell_value(r, c)
                if cell in ("", None):
                    continue
                records.append((date, keys[c], float(cell)))
    out = pd.DataFrame(records, columns=["date", "source_key", "value"])
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    return out.sort_values(["source_key", "date"], ignore_index=True)


def parse_fred_csv(content: bytes, source_key: str) -> pd.DataFrame:
    """Parse a FRED ``fredgraph.csv`` download (missing values are ``.`` or empty)."""
    frame = pd.read_csv(io.BytesIO(content), na_values=["."])
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    frame = frame.dropna(subset=["value"])
    frame["source_key"] = source_key
    return frame[["date", "source_key", "value"]].reset_index(drop=True)


def futures_long(raw: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Map EIA futures source keys to ``(date, code, contract, settle, source_key)``."""
    key_map = {
        key: (code, k + 1) for code, keys in FUTURES_KEYS.items() for k, key in enumerate(keys)
    }
    sub = raw[raw["source_key"].isin(key_map)].copy()
    sub["code"] = sub["source_key"].map(lambda key: key_map[key][0])
    sub["contract"] = sub["source_key"].map(lambda key: key_map[key][1]).astype("int64")
    sub = sub[sub["date"] <= pd.Timestamp(as_of)]
    sub = sub.rename(columns={"value": "settle"})
    cols = ["date", "code", "contract", "settle", "source_key"]
    return sub[cols].sort_values(["code", "date", "contract"], ignore_index=True)


def series_long(raw: pd.DataFrame, catalogue: tuple[Series, ...], as_of: str) -> pd.DataFrame:
    """Keep the catalogue's series from a parsed workbook and attach their metadata."""
    by_key = {s.source_key: s for s in catalogue}
    sub = raw[raw["source_key"].isin(by_key) & (raw["date"] <= pd.Timestamp(as_of))].copy()
    sub["series_id"] = sub["source_key"].map(lambda key: by_key[key].series_id)
    sub["code"] = sub["source_key"].map(lambda key: by_key[key].code)
    sub["region"] = sub["source_key"].map(lambda key: by_key[key].region)
    sub["unit"] = sub["source_key"].map(lambda key: by_key[key].unit)
    cols = ["date", "series_id", "code", "region", "value", "unit", "source_key"]
    return sub[cols].sort_values(["series_id", "date"], ignore_index=True)


__all__ = [
    "INVENTORY_SERIES",
    "SPOT_SERIES",
    "futures_long",
    "parse_eia_workbook",
    "parse_fred_csv",
    "series_long",
]

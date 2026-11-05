"""Download the public raw files and rebuild the frozen sample in ``data/sample``.

Usage::

    python scripts/fetch_data.py              # download EIA + FRED, rewrite the sample
    python scripts/fetch_data.py --ng-strip   # also refresh the Yahoo NG strip snapshot
    python scripts/fetch_data.py --check      # verify data/sample against MANIFEST.json

Needs the ``[fetch]`` extra (``pip install -e ".[fetch]"``). Raw downloads go
to ``data/raw`` (git-ignored). Every non-futures series is cut at
``SAMPLE_AS_OF`` so that a later download reproduces the same sample, except
for revisions published by the EIA (weekly stocks are occasionally revised):
``--check`` reports any difference through the content hashes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

from energycurves.data.eia import futures_long, parse_eia_workbook, parse_fred_csv, series_long
from energycurves.data.registry import (
    EIA_FUTURES_LAST_DATE,
    EIA_NG_FUTURES_URL,
    EIA_PET_FUTURES_URL,
    EIA_PET_SPOT_URL,
    FX_SERIES,
    INVENTORY_SERIES,
    NG_STRIP_FIRST_MONTH,
    NG_STRIP_LAST_MONTH,
    NG_STRIP_SNAPSHOT_DATE,
    NYMEX_MONTH_CODES,
    RATE_SERIES,
    SAMPLE_AS_OF,
    SPOT_SERIES,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SAMPLE = ROOT / "data" / "sample"
MANIFEST = SAMPLE / "MANIFEST.json"
USER_AGENT = "energycurves-research/1.0 (public data download)"


def content_sha256(frame: pd.DataFrame) -> str:
    """Hash of the CSV serialisation: independent of the parquet writer version."""
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def download(url: str, target: Path) -> dict[str, str]:
    """Download ``url`` to ``target``; return provenance (URL, UTC time, sha256)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    target.write_bytes(payload)
    time.sleep(1.0)  # be polite with the public servers
    return {
        "url": url,
        "extracted_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
    }


def describe(frame: pd.DataFrame, date_col: str = "date") -> dict[str, object]:
    return {
        "rows": len(frame),
        "first_date": str(frame[date_col].min().date()),
        "last_date": str(frame[date_col].max().date()),
        "content_sha256": content_sha256(frame),
    }


def build_eia_and_fred() -> dict[str, dict[str, object]]:
    """Download every EIA/FRED file, write the parquet sample, return manifest entries."""
    provenance: dict[str, dict[str, str]] = {}
    raw_frames: dict[str, pd.DataFrame] = {}
    for name, url in {
        "PET_PRI_FUT_S1_D.xls": EIA_PET_FUTURES_URL,
        "NG_PRI_FUT_S1_D.xls": EIA_NG_FUTURES_URL,
        "PET_PRI_SPT_S1_D.xls": EIA_PET_SPOT_URL,
    }.items():
        provenance[name] = download(url, RAW / name)
        raw_frames[name] = parse_eia_workbook(RAW / name)
    for series in INVENTORY_SERIES:
        name = Path(series.url).name
        provenance[name] = download(series.url, RAW / name)
        raw_frames[name] = parse_eia_workbook(RAW / name)

    entries: dict[str, dict[str, object]] = {}

    futures_raw = pd.concat([raw_frames["PET_PRI_FUT_S1_D.xls"], raw_frames["NG_PRI_FUT_S1_D.xls"]])
    futures = futures_long(futures_raw, EIA_FUTURES_LAST_DATE)
    futures.to_parquet(SAMPLE / "futures_daily.parquet", index=False)
    entries["futures_daily.parquet"] = {
        **describe(futures),
        "description": "NYMEX settlements of contracts 1-4 for CL, NG, RB, HO as published by "
        "the EIA (14:30 ET close). No cleaning applied.",
        "units": {"CL": "USD/bbl", "NG": "USD/MMBtu", "RB": "USD/gal", "HO": "USD/gal"},
        "sources": [provenance["PET_PRI_FUT_S1_D.xls"], provenance["NG_PRI_FUT_S1_D.xls"]],
        "source_keys": sorted(futures["source_key"].unique().tolist()),
    }

    spot_raw = pd.concat([raw_frames["PET_PRI_SPT_S1_D.xls"], raw_frames["NG_PRI_FUT_S1_D.xls"]])
    spot = series_long(spot_raw, SPOT_SERIES, SAMPLE_AS_OF)
    spot.to_parquet(SAMPLE / "spot_daily.parquet", index=False)
    entries["spot_daily.parquet"] = {
        **describe(spot),
        "description": "EIA daily spot prices (FOB).",
        "series": {s.series_id: {"source_key": s.source_key, "unit": s.unit} for s in SPOT_SERIES},
        "sources": [provenance["PET_PRI_SPT_S1_D.xls"], provenance["NG_PRI_FUT_S1_D.xls"]],
    }

    inv_raw = pd.concat([raw_frames[Path(s.url).name] for s in INVENTORY_SERIES])
    inventory = series_long(inv_raw, INVENTORY_SERIES, SAMPLE_AS_OF)
    inventory.to_parquet(SAMPLE / "inventory_weekly.parquet", index=False)
    entries["inventory_weekly.parquet"] = {
        **describe(inventory),
        "description": "EIA weekly inventories (week-ending Friday).",
        "series": {
            s.series_id: {"source_key": s.source_key, "unit": s.unit, "region": s.region}
            for s in INVENTORY_SERIES
        },
        "sources": [provenance[Path(s.url).name] for s in INVENTORY_SERIES],
    }

    for out_name, catalogue in (
        ("rates_daily.parquet", RATE_SERIES),
        ("fx_daily.parquet", FX_SERIES),
    ):
        frames, sources = [], []
        for series in catalogue:
            name = f"{series.source_key}.csv"
            sources.append(download(series.url, RAW / name))
            frame = parse_fred_csv((RAW / name).read_bytes(), series.source_key)
            frame["series_id"] = series.series_id
            frame["unit"] = series.unit
            frames.append(frame[frame["date"] <= pd.Timestamp(SAMPLE_AS_OF)])
        data = pd.concat(frames, ignore_index=True)[
            ["date", "series_id", "value", "unit", "source_key"]
        ]
        data.to_parquet(SAMPLE / out_name, index=False)
        entries[out_name] = {
            **describe(data),
            "description": "; ".join(s.description for s in catalogue) + " (FRED).",
            "units": {s.series_id: s.unit for s in catalogue},
            "sources": sources,
        }
    return entries


def build_ng_strip() -> dict[str, object]:
    """Snapshot of the live NYMEX natural-gas strip on ``NG_STRIP_SNAPSHOT_DATE`` (Yahoo)."""
    import yfinance as yf

    snapshot = pd.Timestamp(NG_STRIP_SNAPSHOT_DATE)
    rows = []
    for month in pd.period_range(NG_STRIP_FIRST_MONTH, NG_STRIP_LAST_MONTH, freq="M"):
        ticker = f"NG{NYMEX_MONTH_CODES[month.month - 1]}{month.year % 100:02d}.NYM"
        hist = yf.Ticker(ticker).history(
            start=(snapshot - pd.Timedelta(days=7)).date().isoformat(),
            end=(snapshot + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=False,
        )
        hist.index = pd.DatetimeIndex(hist.index.date)
        if snapshot not in hist.index:
            raise RuntimeError(f"{ticker}: no close on {snapshot.date()}")
        rows.append(
            {
                "snapshot_date": snapshot,
                "code": "NG",
                "delivery_month": month.to_timestamp(),
                "ticker": ticker,
                "settle": float(hist.loc[snapshot, "Close"]),
                "volume": float(hist.loc[snapshot, "Volume"]),
            }
        )
        time.sleep(1.5)
    strip = pd.DataFrame(rows)
    strip.to_parquet(SAMPLE / "ng_strip_snapshot.parquet", index=False)
    return {
        **describe(strip, "delivery_month"),
        "description": "Daily close of every live NYMEX Henry Hub contract on the snapshot date, "
        "from Yahoo Finance via yfinance (close price, not the official CME settlement).",
        "units": {"NG": "USD/MMBtu"},
        "snapshot_date": NG_STRIP_SNAPSHOT_DATE,
        "sources": [
            {
                "url": "https://finance.yahoo.com (tickers NG<month><yy>.NYM)",
                "extracted_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            }
        ],
    }


def check() -> int:
    """Compare every sample file with the content hash recorded in the manifest."""
    manifest = json.loads(MANIFEST.read_text())
    failures = 0
    for name, entry in manifest["files"].items():
        frame = pd.read_parquet(SAMPLE / name)
        ok = content_sha256(frame) == entry["content_sha256"]
        failures += not ok
        print(f"{'OK ' if ok else 'BAD'} {name}  rows={len(frame)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify the sample hashes only")
    parser.add_argument("--ng-strip", action="store_true", help="refresh the Yahoo NG snapshot")
    args = parser.parse_args()
    if args.check:
        return 1 if check() else 0

    SAMPLE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": {}}
    manifest["files"].update(build_eia_and_fred())
    if args.ng_strip or "ng_strip_snapshot.parquet" not in manifest["files"]:
        manifest["files"]["ng_strip_snapshot.parquet"] = build_ng_strip()
    manifest["sample_as_of"] = SAMPLE_AS_OF
    manifest["eia_futures_last_date"] = EIA_FUTURES_LAST_DATE
    manifest["files"] = dict(sorted(manifest["files"].items()))
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return check()


if __name__ == "__main__":
    sys.exit(main())

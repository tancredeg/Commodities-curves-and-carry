# Data

Public sources only. Every file in `data/sample/` is versioned, with its URL, extraction time
and hashes in `data/sample/MANIFEST.json`; `python scripts/fetch_data.py --check` verifies the
content hashes and runs in CI.

## Coverage of the frozen sample

<!-- BEGIN:data-coverage -->
| File | Rows | First | Last | Extracted (UTC) |
| :--- | ---: | :--- | :--- | :--- |
| `futures_daily.parquet` | 121,719 | 1980-01-02 | 2024-04-05 | 2026-11-05 |
| `fx_daily.parquet` | 6,979 | 1999-01-04 | 2026-10-30 | 2026-11-05 |
| `inventory_weekly.parquet` | 7,402 | 1982-08-20 | 2026-10-30 | 2026-11-05 |
| `ng_strip_snapshot.parquet` | 25 | 2026-12-01 | 2028-12-01 | 2026-11-05 |
| `rates_daily.parquet` | 18,201 | 1954-01-04 | 2026-11-03 | 2026-11-05 |
| `spot_daily.parquet` | 47,488 | 1986-01-02 | 2026-11-03 | 2026-11-05 |
<!-- END:data-coverage -->

## Sources

| File | Source | Series | Unit |
| :--- | :--- | :--- | :--- |
| `futures_daily.parquet` | EIA, *NYMEX Futures Prices* (`PET_PRI_FUT_S1_D.xls`) and *Natural Gas Spot and Futures Prices* (`NG_PRI_FUT_S1_D.xls`) | Contracts 1–4 of WTI (`RCLC1-4`), Henry Hub (`RNGC1-4`), RBOB (`EER_EPMRR_PE1-4_Y35NY_DPG`), No. 2 heating oil (`EER_EPD2F_PE1-4_Y35NY_DPG`); official 14:30 ET NYMEX closes | USD/bbl (CL), USD/MMBtu (NG), USD/gal (RB, HO) |
| `spot_daily.parquet` | EIA, *Spot Prices* (`PET_PRI_SPT_S1_D.xls`) and the NG file above | WTI Cushing, Brent, NYH conventional gasoline, NYH heating oil, NYH ULSD, Henry Hub | USD/bbl, USD/gal, USD/MMBtu |
| `inventory_weekly.parquet` | EIA weekly history files | Lower-48 working gas, US crude excl. SPR, US total gasoline, US distillate | Bcf, kbbl |
| `rates_daily.parquet` | FRED `DTB3` | 3-month T-bill, secondary market, discount basis | percent |
| `fx_daily.parquet` | FRED `DEXUSEU` | USD per EUR, noon buying rate | USD per EUR |
| `ng_strip_snapshot.parquet` | Yahoo Finance via `yfinance`, tickers `NG<month><yy>.NYM` | Close of every live Henry Hub contract on the snapshot date | USD/MMBtu |

## Known breaks and choices

- **The EIA stopped publishing NYMEX futures after 2024-04-05** ("Futures prices after April 5,
  2024, are not available"). Every futures-based result therefore ends on that date; spot,
  inventory, rate and FX series are cut at the sample freeze date instead.
- **Units.** RB and HO are quoted in USD/gal; ×42 gives USD/bbl (crack spreads).
- **WTI contract 1 settled at −37.63 USD/bbl on 2020-04-20.** Logs of non-positive prices are
  undefined: roll yields fall back to contracts 2–3 that day and positions are held in
  contract 2, which stayed positive.
- **Heating oil became ULSD.** From the May 2013 contract the NYMEX heating oil future
  references ultra-low-sulfur diesel; the EIA series keeps its name. The series is used as one
  contract across the break.
- **RBOB replaced reformulated gasoline** (the RFG series ends in 2006); RBOB futures start in
  2005, which sets the common start of the four-contract universe.
- **Expiry rules** (`src/energycurves/data/calendar.py`): CL 3 business days before the 25th of
  the month before delivery; NG 3 business days before the first day of the delivery month
  (6 and 5 business days in the mid-1990s, and four 2008–2010 holidays not counted, both found
  in the prices); RB and HO on the last business day of the month before delivery. The
  calendar is checked against the roll days implied by the prices (`tests/test_calendar.py`).
- **Cleaning** (`src/energycurves/data/clean.py`, identical in `sql/15_build_curve.sql`): rows
  dated on exchange holidays (stale copies of the previous close) are dropped, and isolated,
  reverting one-day spikes on contracts 2–4 (typing errors) are removed. The raw EIA values are
  kept untouched in `data/sample`.
- **NG strip snapshot.** Yahoo closes, not official CME settlements, for one day. They define
  the forward curve of the storage valuation; the EIA-only seasonal curve is reported next to it
  as a robustness check.
- **Discount rates.** DTB3 is a discount yield (Act/360); it is converted to a continuously
  compounded Act/365 rate, using the last fixing on or before the valuation date.

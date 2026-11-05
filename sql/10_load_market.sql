-- 10_load_market.sql — market data from the frozen sample, idempotent upserts.
--
-- Input: parquet files in the folder held by the DuckDB variable `sample_dir`
-- (set by energycurves.db; from the DuckDB CLI: SET VARIABLE sample_dir = 'data/sample').
-- Every row keeps its provenance in `source` (provider + series key).

-- Futures settlements, contracts 1-4 (EIA, raw).
INSERT INTO fact_price_daily (date, code, contract, settle, volume, source)
SELECT
    CAST(date AS DATE),
    code,
    CAST(contract AS TINYINT),
    settle,
    CAST(NULL AS DOUBLE),
    'EIA:' || source_key
FROM read_parquet(getvariable('sample_dir') || '/futures_daily.parquet')
ON CONFLICT (date, code, contract) DO UPDATE SET
    settle = EXCLUDED.settle,
    volume = EXCLUDED.volume,
    source = EXCLUDED.source;

-- Live natural-gas strip on the snapshot date (Yahoo Finance closes).
INSERT INTO fact_curve_snapshot
    (snapshot_date, code, delivery_month, ticker, settle, volume, source)
SELECT
    CAST(snapshot_date AS DATE),
    code,
    CAST(delivery_month AS DATE),
    ticker,
    settle,
    volume,
    'Yahoo:' || ticker
FROM read_parquet(getvariable('sample_dir') || '/ng_strip_snapshot.parquet')
ON CONFLICT (snapshot_date, code, delivery_month) DO UPDATE SET
    ticker = EXCLUDED.ticker,
    settle = EXCLUDED.settle,
    volume = EXCLUDED.volume,
    source = EXCLUDED.source;

-- Spot prices (EIA).
INSERT INTO fact_spot_daily (date, series_id, code, price, unit, source)
SELECT CAST(date AS DATE), series_id, code, value, unit, 'EIA:' || source_key
FROM read_parquet(getvariable('sample_dir') || '/spot_daily.parquet')
ON CONFLICT (date, series_id) DO UPDATE SET
    code   = EXCLUDED.code,
    price  = EXCLUDED.price,
    unit   = EXCLUDED.unit,
    source = EXCLUDED.source;

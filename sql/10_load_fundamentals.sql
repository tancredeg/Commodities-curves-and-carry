-- 10_load_fundamentals.sql — inventories, rates and FX, idempotent upserts.
--
-- Input: parquet files in the folder held by the DuckDB variable `sample_dir`.
-- Each upsert is followed by a DELETE of the rows that left the sample.

-- Weekly inventories (EIA, week ending on Friday).
INSERT INTO fact_inventory (week_ending, code, region, stock, unit, source)
SELECT CAST(date AS DATE), code, region, value, unit, 'EIA:' || source_key
FROM read_parquet(getvariable('sample_dir') || '/inventory_weekly.parquet')
ON CONFLICT (week_ending, code, region) DO UPDATE SET
    stock  = EXCLUDED.stock,
    unit   = EXCLUDED.unit,
    source = EXCLUDED.source;

DELETE FROM fact_inventory AS t
WHERE NOT EXISTS (
    SELECT 1 FROM read_parquet(getvariable('sample_dir') || '/inventory_weekly.parquet') AS s
    WHERE CAST(s.date AS DATE) = t.week_ending AND s.code = t.code AND s.region = t.region
);

-- 3-month T-bill, discount basis (FRED).
INSERT INTO fact_rate_daily (date, series_id, value, unit, source)
SELECT CAST(date AS DATE), series_id, value, unit, 'FRED:' || source_key
FROM read_parquet(getvariable('sample_dir') || '/rates_daily.parquet')
ON CONFLICT (date, series_id) DO UPDATE SET
    value  = EXCLUDED.value,
    unit   = EXCLUDED.unit,
    source = EXCLUDED.source;

DELETE FROM fact_rate_daily AS t
WHERE NOT EXISTS (
    SELECT 1 FROM read_parquet(getvariable('sample_dir') || '/rates_daily.parquet') AS s
    WHERE CAST(s.date AS DATE) = t.date AND s.series_id = t.series_id
);

-- EUR/USD noon buying rate (FRED), USD per 1 EUR.
INSERT INTO fx_rates (date, pair, rate, source)
SELECT CAST(date AS DATE), series_id, value, 'FRED:' || source_key
FROM read_parquet(getvariable('sample_dir') || '/fx_daily.parquet')
ON CONFLICT (date, pair) DO UPDATE SET
    rate   = EXCLUDED.rate,
    source = EXCLUDED.source;

DELETE FROM fx_rates AS t
WHERE NOT EXISTS (
    SELECT 1 FROM read_parquet(getvariable('sample_dir') || '/fx_daily.parquet') AS s
    WHERE CAST(s.date AS DATE) = t.date AND s.series_id = t.pair
);

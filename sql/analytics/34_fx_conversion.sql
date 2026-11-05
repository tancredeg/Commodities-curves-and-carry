-- 34_fx_conversion.sql
-- Business question: what did WTI cost a European buyer, in EUR per barrel?
-- Inputs : fact_curve (CL contract 2), fx_rates (EURUSD = USD per 1 EUR).
--          Parameters: $start, $end (period [start, end)).
-- Output : one row per month: average WTI contract-2 settlement in USD/bbl and in
--          EUR/bbl, average EURUSD used, number of days.
-- Pitfall avoided: FRED fixings and NYMEX settlements do not share the same
--   holidays. The ASOF join takes the last fixing ON OR BEFORE each trade date
--   (p.trade_date >= f.date): never a future fixing, and no day is dropped
--   when the fixing is missing.
WITH priced AS (
    SELECT
        p.trade_date,
        p.settle         AS usd_per_bbl,
        f.rate           AS eurusd,
        p.settle / f.rate AS eur_per_bbl
    FROM fact_curve AS p
    ASOF JOIN fx_rates AS f
        ON  f.pair = 'EURUSD'
        AND p.trade_date >= f.date
    WHERE p.code = 'CL'
      AND p.contract = 2
      AND p.trade_date >= $start
      AND p.trade_date <  $end
)
SELECT
    date_trunc('month', trade_date) AS month,
    avg(usd_per_bbl)                AS wti_usd_per_bbl,
    avg(eur_per_bbl)                AS wti_eur_per_bbl,
    avg(eurusd)                     AS eurusd,
    count(*)                        AS n_days
FROM priced
GROUP BY date_trunc('month', trade_date)
ORDER BY month;

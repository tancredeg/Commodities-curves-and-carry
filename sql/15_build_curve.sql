-- 15_build_curve.sql — clean curve with delivery months (full refresh, idempotent).
--
-- Inputs : fact_price_daily (raw EIA slots), dim_business_day, dim_contract_calendar.
-- Output : fact_curve, one row per (trade_date, code, contract) with its delivery
--          month, last trading day and time to expiry.
-- Same rules as energycurves.data.stitch.build_curve_panel (tests check equality):
--   1. drop rows dated on non-business days (stale copies published by the EIA);
--   2. contract k on date t = k-th delivery month whose last trading day is >= t;
--   3. drop isolated one-day spikes on contracts 2-4 (typing errors).

DELETE FROM fact_curve;

INSERT INTO fact_curve
    (trade_date, code, contract, delivery_month, last_trade_date, tau, settle, source)
WITH on_business_days AS (
    -- Rule 1: SEMI JOIN keeps a price row only if its date is a business day,
    -- without duplicating it.
    SELECT p.date, p.code, p.contract, p.settle, p.source
    FROM fact_price_daily AS p
    SEMI JOIN dim_business_day AS b ON b.date = p.date
),
live_months AS (
    -- Rule 2: rank the delivery months still trading on each date. The upper
    -- bound on delivery_month only limits the range join (slot 4 is always
    -- less than 8 months ahead).
    SELECT
        d.date,
        d.code,
        c.delivery_month,
        c.last_trade_date,
        ROW_NUMBER() OVER (
            PARTITION BY d.date, d.code
            ORDER BY c.delivery_month
        ) AS slot
    FROM (SELECT DISTINCT date, code FROM on_business_days) AS d
    JOIN dim_contract_calendar AS c
        ON  c.code = d.code
        AND c.last_trade_date >= d.date
        AND c.delivery_month < d.date + INTERVAL 8 MONTH
),
assigned AS (
    SELECT p.date, p.code, p.contract, l.delivery_month, l.last_trade_date, p.settle, p.source
    FROM on_business_days AS p
    JOIN live_months AS l
        ON l.date = p.date AND l.code = p.code AND l.slot = p.contract
),
log_returns AS (
    -- Same-contract log returns today (r) and tomorrow (r_next). ln() is only
    -- taken on positive prices: the negative WTI settle stays NULL here.
    SELECT
        *,
        lp - LAG(lp)  OVER same_contract AS r,
        LEAD(lp) OVER same_contract - lp AS r_next
    FROM (
        SELECT *, CASE WHEN settle > 0 THEN ln(settle) END AS lp
        FROM assigned
    )
    WINDOW same_contract AS (PARTITION BY code, delivery_month ORDER BY date)
),
spike_flags AS (
    -- Rule 3: a big move reversed the next day while the rest of the curve
    -- is quiet. The median is taken over all contracts of the same day.
    SELECT
        *,
        COALESCE(
            contract >= 2
            AND abs(r) > 0.07
            AND abs(r_next) > 0.07
            AND sign(r) <> sign(r_next)
            AND abs(r + r_next) < 0.3 * abs(r)
            AND median(abs(r)) OVER (PARTITION BY code, date) < 0.3 * abs(r),
            FALSE
        ) AS is_spike
    FROM log_returns
)
SELECT
    date AS trade_date,
    code,
    contract,
    delivery_month,
    last_trade_date,
    datediff('day', date, last_trade_date) / 365.0 AS tau,
    settle,
    source
FROM spike_flags
WHERE NOT is_spike;   -- filter applied after the window functions

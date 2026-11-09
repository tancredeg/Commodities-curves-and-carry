-- 31_spread_zscore.sql
-- Business question: are refining margins (crack spreads) rich or cheap
--   relative to their last 60 sessions?
-- Inputs : fact_curve (CL, RB, HO), dim_commodity (USD/gal -> USD/bbl factor).
--          Parameters: $start, $end (display period [start, end)).
-- Output : one row per spread and trade date: delivery month shown (CL contract 2),
--          spread in USD/bbl, 60-session mean (back-adjusted, in today's contract
--          terms) and standard deviation, z60 (window ends today, descriptive) and
--          z60_exec (window ends yesterday: usable by a signal executed at today's
--          close). Spreads: crack_321 = (2 RB + 1 HO)/3 - CL, gasoline crack RB - CL,
--          distillate crack HO - CL.
-- Pitfalls avoided: (1) the legs are matched on DELIVERY MONTH, not on slot
--   number (CL expires ~10 days before RB/HO, so "contract 2" is not the same
--   month for all three); (2) USD/gal is converted to USD/bbl with the factor
--   stored in dim_commodity; (3) the shown month changes at each CL expiry and
--   crack spreads are seasonal, so the window runs on a BACK-ADJUSTED series:
--   each daily change is measured on one delivery month (yesterday's), then
--   cumulated; a change of month never enters the mean, the deviation or the
--   z-score; (4) windows run on the full history and the date filter comes
--   last, so the first displayed z-scores use 60 real sessions.
WITH legs AS (
    SELECT
        c.trade_date,
        c.delivery_month,
        c.code,
        c.contract,
        c.settle * d.usd_per_bbl_factor AS usd_per_bbl
    FROM fact_curve AS c
    JOIN dim_commodity AS d ON d.code = c.code
    WHERE c.code IN ('CL', 'RB', 'HO')
),
matched AS (
    -- Every delivery month whose three legs settle that day, whatever the slot.
    SELECT
        cl.trade_date,
        cl.delivery_month,
        cl.contract    AS cl_contract,
        cl.usd_per_bbl AS cl,
        rb.usd_per_bbl AS rb,
        ho.usd_per_bbl AS ho
    FROM legs AS cl
    JOIN legs AS rb
        ON  rb.code = 'RB'
        AND rb.trade_date = cl.trade_date
        AND rb.delivery_month = cl.delivery_month
    JOIN legs AS ho
        ON  ho.code = 'HO'
        AND ho.trade_date = cl.trade_date
        AND ho.delivery_month = cl.delivery_month
    WHERE cl.code = 'CL'
),
spreads AS (
    SELECT trade_date, delivery_month, cl_contract, 'crack_321' AS spread,
           (2.0 * rb + ho) / 3.0 - cl AS value
    FROM matched
    UNION ALL
    SELECT trade_date, delivery_month, cl_contract, 'gasoline_crack', rb - cl FROM matched
    UNION ALL
    SELECT trade_date, delivery_month, cl_contract, 'distillate_crack', ho - cl FROM matched
),
shown AS (
    -- The month shown is the CL contract 2 month; prev_* = previous shown session.
    SELECT
        spread,
        trade_date,
        delivery_month,
        value,
        LAG(trade_date)     OVER by_spread AS prev_date,
        LAG(delivery_month) OVER by_spread AS prev_month
    FROM spreads
    WHERE cl_contract = 2
    WINDOW by_spread AS (PARTITION BY spread ORDER BY trade_date)
),
adjusted AS (
    -- Back-adjusted level: cumulated changes of yesterday's month, measured on
    -- that same month (0 if a leg is missing).
    SELECT
        s.spread,
        s.trade_date,
        s.delivery_month,
        s.value,
        sum(coalesce(now.value - before.value, 0.0)) OVER (
            PARTITION BY s.spread ORDER BY s.trade_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS level
    FROM shown AS s
    LEFT JOIN spreads AS now
        ON  now.spread = s.spread
        AND now.trade_date = s.trade_date
        AND now.delivery_month = s.prev_month
    LEFT JOIN spreads AS before
        ON  before.spread = s.spread
        AND before.trade_date = s.prev_date
        AND before.delivery_month = s.prev_month
),
windowed AS (
    SELECT
        spread,
        trade_date,
        delivery_month,
        value,
        level,
        avg(level)         OVER last_60     AS level_mean_60,
        stddev_samp(level) OVER last_60     AS sd_60,
        avg(level)         OVER previous_60 AS level_mean_60_prev,
        stddev_samp(level) OVER previous_60 AS sd_60_prev,
        count(level)       OVER previous_60 AS n_prev
    FROM adjusted
    WINDOW
        last_60 AS (
            PARTITION BY spread ORDER BY trade_date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ),
        previous_60 AS (
            PARTITION BY spread ORDER BY trade_date
            ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
        )
)
SELECT
    spread,
    trade_date,
    delivery_month,
    value,
    value - (level - level_mean_60) AS mean_60,
    sd_60,
    (level - level_mean_60) / nullif(sd_60, 0) AS z60,
    CASE
        WHEN n_prev = 60 THEN (level - level_mean_60_prev) / nullif(sd_60_prev, 0)
    END AS z60_exec
FROM windowed
WHERE trade_date >= $start
  AND trade_date <  $end
ORDER BY spread, trade_date;

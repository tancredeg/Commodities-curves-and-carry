-- 31_spread_zscore.sql
-- Business question: are refining margins (crack spreads) rich or cheap
--   relative to their last 60 sessions?
-- Inputs : fact_curve (CL, RB, HO), dim_commodity (USD/gal -> USD/bbl factor).
--          Parameters: $start, $end (display period [start, end)).
-- Output : one row per spread and trade date: delivery month, spread in USD/bbl,
--          60-session mean and standard deviation, z60 (window ends today,
--          descriptive) and z60_exec (window ends yesterday: usable by a signal
--          executed at today's close). Spreads: crack_321 = (2 RB + 1 HO)/3 - CL,
--          gasoline crack RB - CL, distillate crack HO - CL.
-- Pitfalls avoided: (1) the legs are matched on DELIVERY MONTH, not on slot
--   number (CL expires ~10 days before RB/HO, so "contract 2" is not the same
--   month for all three); (2) USD/gal is converted to USD/bbl with the factor
--   stored in dim_commodity; (3) windows run on the full history and the date
--   filter comes last, so the first displayed z-scores use 60 real sessions.
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
    -- CL contract 2 fixes the delivery month; RB and HO for that same month.
    SELECT
        cl.trade_date,
        cl.delivery_month,
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
      AND cl.contract = 2
),
spreads AS (
    SELECT trade_date, delivery_month, 'crack_321' AS spread,
           (2.0 * rb + ho) / 3.0 - cl AS value
    FROM matched
    UNION ALL
    SELECT trade_date, delivery_month, 'gasoline_crack', rb - cl FROM matched
    UNION ALL
    SELECT trade_date, delivery_month, 'distillate_crack', ho - cl FROM matched
),
windowed AS (
    SELECT
        spread,
        trade_date,
        delivery_month,
        value,
        avg(value)         OVER last_60     AS mean_60,
        stddev_samp(value) OVER last_60     AS sd_60,
        avg(value)         OVER previous_60 AS mean_60_prev,
        stddev_samp(value) OVER previous_60 AS sd_60_prev,
        count(value)       OVER previous_60 AS n_prev
    FROM spreads
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
    mean_60,
    sd_60,
    (value - mean_60) / nullif(sd_60, 0) AS z60,
    CASE
        WHEN n_prev = 60 THEN (value - mean_60_prev) / nullif(sd_60_prev, 0)
    END AS z60_exec
FROM windowed
WHERE trade_date >= $start
  AND trade_date <  $end
ORDER BY spread, trade_date;

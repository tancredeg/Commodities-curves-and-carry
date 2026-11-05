-- 30_term_structure.sql
-- Business question: how often is each curve in backwardation or contango, and
--   what annualised roll yield does the front of the curve offer on average?
-- Inputs : v_roll_yield (fact_curve). Parameters: $start, $end (period [start, end)),
--          $band (dead band on |y12|, per year: |y12| <= band counts as flat).
-- Output : one row per commodity: number of days, mean / median y12, share of days
--          in backwardation / contango / flat, mean y14 (contracts 1 to 4).
-- Pitfall avoided: the slope uses two prices of the SAME day divided by their exact
--   time-to-expiry gap, never a return across a roll; a dead band keeps nearly
--   flat curves from being counted as backwardation or contango.
WITH daily AS (
    SELECT
        code,
        trade_date,
        y12,
        y14,
        CASE
            WHEN y12 >  $band THEN 'backwardation'
            WHEN y12 < -$band THEN 'contango'
            ELSE 'flat'
        END AS regime
    FROM v_roll_yield
    WHERE trade_date >= $start
      AND trade_date <  $end
      AND y12 IS NOT NULL
)
SELECT
    code,
    count(*)                                                    AS n_days,
    min(trade_date)                                             AS first_date,
    max(trade_date)                                             AS last_date,
    avg(y12)                                                    AS mean_y12,
    median(y12)                                                 AS median_y12,
    avg(CASE WHEN regime = 'backwardation' THEN 1.0 ELSE 0.0 END) AS share_backwardation,
    avg(CASE WHEN regime = 'contango'      THEN 1.0 ELSE 0.0 END) AS share_contango,
    avg(CASE WHEN regime = 'flat'          THEN 1.0 ELSE 0.0 END) AS share_flat,
    avg(y14)                                                    AS mean_y14
FROM daily
GROUP BY code
ORDER BY code;

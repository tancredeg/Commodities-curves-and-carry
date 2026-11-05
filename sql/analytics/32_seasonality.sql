-- 32_seasonality.sql
-- Business question: which delivery months carry a seasonal premium? For natural
--   gas and products the front slope changes sign with the season; a carry
--   signal that ignores this is a seasonal bet in disguise.
-- Inputs : v_roll_yield. Parameters: $start, $end (period [start, end)).
-- Output : one row per commodity and calendar month of the front contract (m1):
--          days, mean and standard deviation of y12, share of days in
--          backwardation (y12 > 0).
-- Pitfall avoided: grouping by the delivery month of contract 1 (m1), not by the
--   trade date month: around expiry the trade month and the front month differ.
SELECT
    code,
    month(m1)                                         AS front_month,
    count(*)                                          AS n_days,
    avg(y12)                                          AS mean_y12,
    stddev_samp(y12)                                  AS sd_y12,
    avg(CASE WHEN y12 > 0 THEN 1.0 ELSE 0.0 END)      AS share_backwardation
FROM v_roll_yield
WHERE trade_date >= $start
  AND trade_date <  $end
  AND y12 IS NOT NULL
GROUP BY code, month(m1)
ORDER BY code, front_month;

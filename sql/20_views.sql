-- 20_views.sql — reusable views on the clean curve.

-- Same-contract daily returns: LAG is partitioned by delivery month, so a
-- return never compares two different contracts (no roll jump).
CREATE OR REPLACE VIEW v_daily_returns AS
SELECT
    trade_date,
    code,
    contract,
    delivery_month,
    settle,
    LAG(trade_date) OVER same_contract AS prev_trade_date,
    CASE
        WHEN LAG(settle) OVER same_contract > 0
        THEN settle / LAG(settle) OVER same_contract - 1
    END AS ret
FROM fact_curve
WINDOW same_contract AS (PARTITION BY code, delivery_month ORDER BY trade_date);

-- Return of a position held in slot k from the previous trading day of the
-- commodity to today. If contract 1 expired yesterday the held contract has
-- moved to slot k-1 today: `rolled` is TRUE and the position is re-established
-- in slot k at today's close. Same logic as energycurves.data.stitch.slot_returns.
CREATE OR REPLACE VIEW v_slot_returns AS
WITH trading_days AS (
    SELECT
        code,
        trade_date,
        LAG(trade_date) OVER (PARTITION BY code ORDER BY trade_date) AS prev_trade_date
    FROM (SELECT DISTINCT code, trade_date FROM fact_curve)
)
SELECT
    d.trade_date,
    d.code,
    held.contract                         AS slot,
    held.delivery_month                   AS held_month,
    CASE WHEN held.settle > 0 THEN today.settle / held.settle - 1 END AS ret,
    today.contract IS DISTINCT FROM held.contract AS rolled
FROM trading_days AS d
JOIN fact_curve AS held                    -- one row per slot held yesterday
    ON  held.code = d.code
    AND held.trade_date = d.prev_trade_date
LEFT JOIN fact_curve AS today              -- same contract, today (may be missing)
    ON  today.code = d.code
    AND today.trade_date = d.trade_date
    AND today.delivery_month = held.delivery_month;

-- One row per trade date and commodity with the four curve points side by
-- side (conditional aggregation with FILTER).
CREATE OR REPLACE VIEW v_curve_points AS
SELECT
    trade_date,
    code,
    max(settle)         FILTER (WHERE contract = 1) AS f1,
    max(settle)         FILTER (WHERE contract = 2) AS f2,
    max(settle)         FILTER (WHERE contract = 3) AS f3,
    max(settle)         FILTER (WHERE contract = 4) AS f4,
    max(tau)            FILTER (WHERE contract = 1) AS tau1,
    max(tau)            FILTER (WHERE contract = 2) AS tau2,
    max(tau)            FILTER (WHERE contract = 3) AS tau3,
    max(tau)            FILTER (WHERE contract = 4) AS tau4,
    max(delivery_month) FILTER (WHERE contract = 1) AS m1
FROM fact_curve
GROUP BY trade_date, code;

-- Annualised roll yields between curve points, y > 0 = backwardation:
--   y(i,j) = (ln F_i − ln F_j) / (tau_j − tau_i)
-- y12 falls back to NULL when F1 <= 0 (WTI, 2020-04-20).
CREATE OR REPLACE VIEW v_roll_yield AS
SELECT
    trade_date,
    code,
    m1,
    CASE WHEN f1 > 0 AND f2 > 0 THEN (ln(f1) - ln(f2)) / (tau2 - tau1) END AS y12,
    CASE WHEN f2 > 0 AND f3 > 0 THEN (ln(f2) - ln(f3)) / (tau3 - tau2) END AS y23,
    CASE WHEN f1 > 0 AND f4 > 0 THEN (ln(f1) - ln(f4)) / (tau4 - tau1) END AS y14
FROM v_curve_points;

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
-- commodity to today. The slot occupant comes from the curve structure, not
-- from the presence of a price (contract c in month m => front month m - (c-1)),
-- so a missing settle neither fakes a roll nor loses the return: p0 is the last
-- settle of the held contract within 5 sessions and since it entered the slot.
-- `rolled` is TRUE when another month occupies the slot today (an expiry in
-- between). Same logic as energycurves.data.stitch.slot_returns.
CREATE OR REPLACE VIEW v_slot_returns AS
WITH days AS (
    SELECT
        code,
        trade_date,
        ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date) AS day_no,
        CAST(min(delivery_month - to_months(contract - 1)) AS DATE) AS front_month
    FROM fact_curve
    GROUP BY code, trade_date
),
slots AS (
    SELECT
        d.code,
        d.trade_date,
        d.day_no,
        s.slot,
        CAST(d.front_month + to_months(s.slot - 1) AS DATE) AS occupant
    FROM days AS d
    CROSS JOIN range(1, 5) AS s(slot)
),
positions AS (                             -- held over (yesterday, today]
    SELECT
        t.code,
        t.trade_date,
        t.day_no,
        y.day_no      AS prev_day_no,
        t.slot,
        y.occupant    AS held_month,
        t.occupant    AS today_occupant
    FROM slots AS t
    JOIN slots AS y
        ON  y.code = t.code
        AND y.slot = t.slot
        AND y.day_no = t.day_no - 1
),
entries AS (                               -- first session of a month in the slot
    SELECT code, slot, occupant, min(day_no) AS entry_day_no
    FROM slots
    GROUP BY code, slot, occupant
),
observed AS (
    SELECT f.code, f.delivery_month, d.day_no, f.settle
    FROM fact_curve AS f
    JOIN days AS d
        ON  d.code = f.code
        AND d.trade_date = f.trade_date
)
SELECT
    p.trade_date,
    p.code,
    p.slot,
    p.held_month,
    CASE
        WHEN p0.settle > 0 AND p0.day_no >= greatest(e.entry_day_no, p.prev_day_no - 4)
        THEN p1.settle / p0.settle - 1
    END AS ret,
    p.today_occupant <> p.held_month AS rolled
FROM positions AS p
JOIN entries AS e
    ON  e.code = p.code
    AND e.slot = p.slot
    AND e.occupant = p.held_month
ASOF LEFT JOIN observed AS p0              -- last settle on or before yesterday
    ON  p0.code = p.code
    AND p0.delivery_month = p.held_month
    AND p.prev_day_no >= p0.day_no
LEFT JOIN observed AS p1                   -- same contract, today (may be missing)
    ON  p1.code = p.code
    AND p1.delivery_month = p.held_month
    AND p1.day_no = p.day_no;

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

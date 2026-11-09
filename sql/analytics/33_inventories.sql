-- 33_inventories.sql
-- Business question: do inventories explain the shape of the curve? The theory
--   of storage (Working, 1949; Fama & French, 1987) predicts contango when stocks
--   are high and backwardation when they are scarce.
-- Inputs : fact_inventory (weekly), v_roll_yield (daily).
--          Parameters: $start, $end (period [start, end)).
-- Output : one row per commodity and week: stock, deviation from the average of
--          the same ISO week over the 5 previous years (%), change versus the same
--          week one year earlier (%), front slope y12 and y14 on the last trading
--          day up to the week ending, and, per commodity, the correlation and OLS
--          slope of y14 on the 5-year deviation over the displayed period.
-- Pitfalls avoided: (1) the 5-year average uses the 5 previous years only (RANGE
--   frame on the ISO year, so a missing week 53 cannot stretch the window, and
--   no future stock enters); (2) the curve is attached with an ASOF join (last
--   trade date <= week ending): a holiday Friday still finds a curve and no
--   later price leaks in; (3) one row per week and region (primary key) before
--   the windows, so the frames count years, not duplicates.
WITH weekly AS (
    SELECT
        code,
        region,
        week_ending,
        stock,
        isoyear(week_ending) AS iso_year,
        week(week_ending)    AS iso_week
    FROM fact_inventory
),
with_history AS (
    SELECT
        code,
        region,
        week_ending,
        stock,
        iso_year,
        iso_week,
        avg(stock) OVER same_week_5y AS avg_5y,
        count(*)   OVER same_week_5y AS n_years,
        lag(stock) OVER same_week_1y AS stock_1y_ago,
        lag(iso_year) OVER same_week_1y AS year_1y_ago
    FROM weekly
    WINDOW
        same_week_5y AS (
            PARTITION BY code, region, iso_week
            ORDER BY iso_year
            RANGE BETWEEN 5 PRECEDING AND 1 PRECEDING   -- the 5 previous YEARS
        ),
        same_week_1y AS (
            PARTITION BY code, region, iso_week
            ORDER BY iso_year
        )
),
with_curve AS (
    SELECT
        h.code,
        h.region,
        h.week_ending,
        h.stock,
        100.0 * (h.stock / h.avg_5y - 1)                                     AS dev_5y_pct,
        CASE WHEN h.year_1y_ago = h.iso_year - 1
             THEN 100.0 * (h.stock / h.stock_1y_ago - 1) END                 AS yoy_pct,
        r.trade_date                                                         AS curve_date,
        r.y12,
        r.y14
    FROM with_history AS h
    ASOF JOIN v_roll_yield AS r
        ON  r.code = h.code
        AND h.week_ending >= r.trade_date
    WHERE h.n_years = 5
      AND h.week_ending >= $start
      AND h.week_ending <  $end
)
SELECT
    code,
    region,
    week_ending,
    stock,
    dev_5y_pct,
    yoy_pct,
    curve_date,
    y12,
    y14,
    corr(y14, dev_5y_pct)        OVER (PARTITION BY code, region) AS corr_y14_dev,
    regr_slope(y14, dev_5y_pct)  OVER (PARTITION BY code, region) AS slope_y14_dev,
    count(y14)                   OVER (PARTITION BY code, region) AS n_weeks
FROM with_curve
ORDER BY code, week_ending;

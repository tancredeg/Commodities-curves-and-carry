-- 10_load_reference.sql — reference tables, idempotent upserts.
--
-- Inputs registered by energycurves.db from the Python catalogue and calendar:
--   src_commodity, src_business_day, src_contract_calendar.
-- Each upsert is followed by a DELETE of the rows that left the source: upserts
-- alone never delete, so a rebuild on an existing file would keep stale rows.

INSERT INTO dim_commodity
    (code, name, unit, currency, point_value, point_unit, usd_per_bbl_factor, expiry_rule)
SELECT code, name, unit, 'USD', point_value, point_unit, usd_per_bbl_factor, expiry_rule
FROM src_commodity
ON CONFLICT (code) DO UPDATE SET
    name               = EXCLUDED.name,
    unit               = EXCLUDED.unit,
    currency           = EXCLUDED.currency,
    point_value        = EXCLUDED.point_value,
    point_unit         = EXCLUDED.point_unit,
    usd_per_bbl_factor = EXCLUDED.usd_per_bbl_factor,
    expiry_rule        = EXCLUDED.expiry_rule;

DELETE FROM dim_commodity AS t
WHERE NOT EXISTS (SELECT 1 FROM src_commodity AS s WHERE s.code = t.code);

INSERT INTO dim_business_day (date)
SELECT CAST(date AS DATE) FROM src_business_day
ON CONFLICT (date) DO NOTHING;

DELETE FROM dim_business_day AS t
WHERE NOT EXISTS (SELECT 1 FROM src_business_day AS s WHERE CAST(s.date AS DATE) = t.date);

INSERT INTO dim_contract_calendar (code, delivery_month, last_trade_date)
SELECT code, CAST(delivery_month AS DATE), CAST(last_trade_date AS DATE)
FROM src_contract_calendar
ON CONFLICT (code, delivery_month) DO UPDATE SET
    last_trade_date = EXCLUDED.last_trade_date;

DELETE FROM dim_contract_calendar AS t
WHERE NOT EXISTS (
    SELECT 1 FROM src_contract_calendar AS s
    WHERE s.code = t.code AND CAST(s.delivery_month AS DATE) = t.delivery_month
);

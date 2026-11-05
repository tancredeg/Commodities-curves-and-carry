-- 10_load_reference.sql — reference tables, idempotent upserts.
--
-- Inputs registered by energycurves.db from the Python catalogue and calendar:
--   src_commodity, src_business_day, src_contract_calendar.

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

INSERT INTO dim_business_day (date)
SELECT CAST(date AS DATE) FROM src_business_day
ON CONFLICT (date) DO NOTHING;

INSERT INTO dim_contract_calendar (code, delivery_month, last_trade_date)
SELECT code, CAST(delivery_month AS DATE), CAST(last_trade_date AS DATE)
FROM src_contract_calendar
ON CONFLICT (code, delivery_month) DO UPDATE SET
    last_trade_date = EXCLUDED.last_trade_date;

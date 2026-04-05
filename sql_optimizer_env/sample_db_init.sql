-- ─────────────────────────────────────────────────────────────────────────────
-- Sample Database for Local Development and Testing
--
-- This is NOT part of the environment itself.
-- It exists only so developers can test the agent locally
-- without needing their own production database.
--
-- Schema deliberately has slow query patterns baked in:
--   - Large tables (500K+ rows) so seq scans are actually slow
--   - Skewed data distributions so postgres stats go wrong
--   - No composite indexes so the agent has room to help
--   - Correlated subquery patterns in the sample queries
--
-- REQUIREMENT:
--   pg_hint_plan must be loaded for index/join hints to work.
--   This init script loads it automatically if available.
-- ─────────────────────────────────────────────────────────────────────────────

-- Load pg_hint_plan if available
-- If not installed, hint actions will still rewrite the SQL correctly
-- but Postgres will ignore the hint comments
DO $$
BEGIN
    BEGIN
        CREATE EXTENSION IF NOT EXISTS pg_hint_plan;
        RAISE NOTICE 'pg_hint_plan loaded successfully';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'pg_hint_plan not available — hint actions will have no effect on this DB';
    END;
END $$;

-- ── Tables ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS regions (
    id          SERIAL PRIMARY KEY,
    region_name VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
    id         SERIAL PRIMARY KEY,
    name       VARCHAR(100) NOT NULL,
    email      VARCHAR(150),
    country    VARCHAR(50),
    region_id  INT REFERENCES regions(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS orders (
    order_id   SERIAL PRIMARY KEY,
    cust_id    INT REFERENCES customers(id),
    status     VARCHAR(20),
    amount     DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS valid_statuses (
    status VARCHAR(20) PRIMARY KEY,
    active BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id    SERIAL PRIMARY KEY,
    order_id   INT REFERENCES orders(order_id),
    product    VARCHAR(100),
    quantity   INT,
    unit_price DECIMAL(10,2)
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
-- All indexes are created upfront.
-- The agent hints which ones to USE — it never creates new ones.

CREATE INDEX IF NOT EXISTS idx_orders_status
    ON orders(status);

CREATE INDEX IF NOT EXISTS idx_orders_cust_id
    ON orders(cust_id);

CREATE INDEX IF NOT EXISTS idx_orders_amount
    ON orders(amount);

CREATE INDEX IF NOT EXISTS idx_orders_created_at
    ON orders(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_customers_country
    ON customers(country);

CREATE INDEX IF NOT EXISTS idx_customers_region_id
    ON customers(region_id);

CREATE INDEX IF NOT EXISTS idx_order_items_order_id
    ON order_items(order_id);

-- ── Seed data ─────────────────────────────────────────────────────────────────
-- Deliberately skewed distributions so postgres stats are often wrong:
--   - 95% of orders are 'closed', only 0.5% are 'open'
--   - Only 5% of customers are from 'US'
-- This makes the agent's index hints and predicate pushes actually matter.

-- Regions (small lookup table — 50 rows)
INSERT INTO regions (region_name)
SELECT 'Region ' || i
FROM generate_series(1, 50) i
ON CONFLICT DO NOTHING;

-- Customers (500K rows)
-- Only 5% are US — postgres will often overestimate and skip indexes
INSERT INTO customers (name, email, country, region_id)
SELECT
    'Customer ' || i,
    'customer' || i || '@example.com',
    CASE
        WHEN random() < 0.05 THEN 'US'
        WHEN random() < 0.15 THEN 'UK'
        WHEN random() < 0.25 THEN 'DE'
        ELSE 'OTHER'
    END,
    (random() * 49 + 1)::int
FROM generate_series(1, 500000) i
ON CONFLICT DO NOTHING;

-- Orders (2M rows)
-- Only ~1% are 'open' — huge opportunity for index hints
INSERT INTO orders (cust_id, status, amount)
SELECT
    (random() * 499999 + 1)::int,
    CASE
        WHEN random() < 0.01  THEN 'open'
        WHEN random() < 0.05  THEN 'pending'
        ELSE 'closed'
    END,
    round((random() * 1000)::numeric, 2)
FROM generate_series(1, 2000000) i
ON CONFLICT DO NOTHING;

-- Valid statuses (tiny lookup table — 3 rows)
-- Used in correlated subquery patterns
INSERT INTO valid_statuses (status, active) VALUES
    ('open',    true),
    ('pending', true),
    ('closed',  false)
ON CONFLICT DO NOTHING;

-- Order items (5M rows — makes SELECT * very expensive)
INSERT INTO order_items (order_id, product, quantity, unit_price)
SELECT
    (random() * 1999999 + 1)::int,
    'Product ' || (random() * 1000)::int,
    (random() * 10 + 1)::int,
    round((random() * 500)::numeric, 2)
FROM generate_series(1, 5000000) i
ON CONFLICT DO NOTHING;

-- ── Update statistics ─────────────────────────────────────────────────────────
-- Run ANALYZE so postgres has fresh stats after seeding.
-- During testing you can manually make stats stale by NOT running this —
-- that will trigger the stale-stats scenario where index hints help most.
ANALYZE regions;
ANALYZE customers;
ANALYZE orders;
ANALYZE valid_statuses;
ANALYZE order_items;

-- ── Confirmation ──────────────────────────────────────────────────────────────
DO $$
DECLARE
    customer_count  BIGINT;
    order_count     BIGINT;
    item_count      BIGINT;
BEGIN
    SELECT COUNT(*) INTO customer_count FROM customers;
    SELECT COUNT(*) INTO order_count    FROM orders;
    SELECT COUNT(*) INTO item_count     FROM order_items;

    RAISE NOTICE '─────────────────────────────────────────';
    RAISE NOTICE 'Sample DB ready:';
    RAISE NOTICE '  customers:   %', customer_count;
    RAISE NOTICE '  orders:      %', order_count;
    RAISE NOTICE '  order_items: %', item_count;
    RAISE NOTICE '─────────────────────────────────────────';
    RAISE NOTICE 'Connect string:';
    RAISE NOTICE '  postgresql://postgres:postgres@localhost:5432/sample';
    RAISE NOTICE '─────────────────────────────────────────';
END $$;
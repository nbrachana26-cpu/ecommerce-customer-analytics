/* =========================================================
   E-COMMERCE ANALYTICS — SQL BUSINESS QUESTIONS
   Database: data/ecommerce.db (SQLite)
   Tables: customers, products, orders, order_items
   =========================================================
   Each query answers a real business question a stakeholder
   would ask. Run with: sqlite3 data/ecommerce.db < business_questions.sql
   ========================================================= */


-- 1. Monthly revenue trend (for exec dashboard)
SELECT
    strftime('%Y-%m', order_date) AS month,
    ROUND(SUM(order_total), 2)     AS revenue,
    COUNT(DISTINCT order_id)       AS orders,
    COUNT(DISTINCT customer_id)    AS active_customers
FROM orders
GROUP BY month
ORDER BY month;


-- 2. Top 10 customers by lifetime value
SELECT
    c.customer_id,
    c.region,
    c.segment,
    COUNT(o.order_id)        AS total_orders,
    ROUND(SUM(o.order_total), 2) AS lifetime_value
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.customer_id
ORDER BY lifetime_value DESC
LIMIT 10;


-- 3. Revenue and margin by product category
SELECT
    p.category,
    ROUND(SUM(oi.line_total), 2) AS revenue,
    ROUND(SUM(oi.quantity * (p.unit_price - p.unit_cost) * (1 - oi.discount)), 2) AS est_profit,
    COUNT(DISTINCT oi.order_id) AS orders_containing_category
FROM order_items oi
JOIN products p ON p.product_id = oi.product_id
GROUP BY p.category
ORDER BY revenue DESC;


-- 4. Best-selling products by quantity and revenue
SELECT
    p.product_name,
    p.category,
    SUM(oi.quantity)              AS units_sold,
    ROUND(SUM(oi.line_total), 2)  AS revenue
FROM order_items oi
JOIN products p ON p.product_id = oi.product_id
GROUP BY p.product_name
ORDER BY revenue DESC
LIMIT 10;


-- 5. Average order value (AOV) by customer segment and region
SELECT
    c.segment,
    c.region,
    ROUND(AVG(o.order_total), 2) AS avg_order_value,
    COUNT(o.order_id)            AS num_orders
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
GROUP BY c.segment, c.region
ORDER BY avg_order_value DESC;


-- 6. Customer recency — days since each customer's last order
-- (base for churn labeling: anything >120 days = at risk)
SELECT
    customer_id,
    MAX(order_date) AS last_order_date,
    CAST(julianday('2024-12-30') - julianday(MAX(order_date)) AS INTEGER) AS days_since_last_order
FROM orders
GROUP BY customer_id
ORDER BY days_since_last_order DESC
LIMIT 15;


-- 7. Monthly cohort retention (signup month vs. purchase month)
WITH first_purchase AS (
    SELECT customer_id, MIN(strftime('%Y-%m', order_date)) AS cohort_month
    FROM orders
    GROUP BY customer_id
),
activity AS (
    SELECT
        o.customer_id,
        fp.cohort_month,
        strftime('%Y-%m', o.order_date) AS order_month
    FROM orders o
    JOIN first_purchase fp ON fp.customer_id = o.customer_id
)
SELECT
    cohort_month,
    order_month,
    COUNT(DISTINCT customer_id) AS active_customers
FROM activity
GROUP BY cohort_month, order_month
ORDER BY cohort_month, order_month
LIMIT 30;


-- 8. Discount impact — does higher discount correlate with bigger basket size?
SELECT
    oi.discount,
    COUNT(*)                      AS line_items,
    ROUND(AVG(oi.quantity), 2)    AS avg_qty,
    ROUND(AVG(oi.line_total), 2)  AS avg_line_revenue
FROM order_items oi
GROUP BY oi.discount
ORDER BY oi.discount;


-- 9. Repeat purchase rate — % of customers with more than 1 order
SELECT
    ROUND(100.0 * SUM(CASE WHEN order_count > 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS repeat_purchase_rate_pct
FROM (
    SELECT customer_id, COUNT(order_id) AS order_count
    FROM orders
    GROUP BY customer_id
);


-- 10. Seasonality — orders and revenue by calendar month across years
SELECT
    strftime('%m', order_date) AS calendar_month,
    ROUND(SUM(order_total), 2) AS revenue,
    COUNT(*) AS orders
FROM orders
GROUP BY calendar_month
ORDER BY calendar_month;

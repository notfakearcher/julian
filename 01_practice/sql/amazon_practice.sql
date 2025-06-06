-- -----------------------------------------------------------------------------
-- Revenue Calculation per Category / Customer / Product
-- -----------------------------------------------------------------------------

-- Revenue per product
SELECT
  product_id,
  ROUND(SUM(quantity * unit_price * (1 - discount_pct)), 0) AS total_revenue
FROM orders
GROUP BY product_id
ORDER BY product_id;

-- Revenue per customer
SELECT
  customer_id,
  ROUND(SUM(quantity * unit_price * (1 - discount_pct)), 0) AS total_revenue
FROM orders
GROUP BY customer_id
ORDER BY customer_id;

-- Revenue per category (join with products)
SELECT
  p.category,
  ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount_pct)), 0) AS total_revenue
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY p.category
ORDER BY p.category;



-- -----------------------------------------------------------------------------
-- Monthly Active Users (MAU), Daily Active Users (DAU), Retention Logic
-- -----------------------------------------------------------------------------

-- DAU for a given date '2024-06-05'
SELECT
  activity_date,
  COUNT(DISTINCT user_id) AS DAU
FROM user_activity
WHERE activity_date = '2024-06-05'
GROUP BY activity_date;

-- MAU for a given month (June 2024)
SELECT
  DATE_FORMAT(activity_date, '%Y-%M'),
  COUNT(DISTINCT user_id) AS MAU
FROM user_activity
WHERE DATE_FORMAT(activity_date, '%Y-%M') = '2024-June'
GROUP BY DATE_FORMAT(activity_date, '%Y-%M');

-- Retention: users active on day 1 and also active 3 months later
SELECT
  DISTINCT ua1.user_id
FROM user_activity ua1
JOIN user_activity ua7 ON ua1.user_id = ua7.user_id
WHERE 
  DATE_FORMAT(ua1.activity_date, '%Y-%M') = (
   SELECT DATE_FORMAT(MIN(activity_date), '%Y-%M') FROM user_activity
  )
  AND
  DATE_FORMAT(ua7.activity_date, '%Y-%M') = (
   SELECT DATE_FORMAT(DATE_ADD(MIN(activity_date), INTERVAL 3 MONTH), '%Y-%M') FROM user_activity
  );


-- -----------------------------------------------------------------------------
  -- Time-windowed Metrics (Past 7-day revenue, rolling 30-day churn)
-- -----------------------------------------------------------------------------

-- Past 7-day revenue as of '2024-06-05'
SELECT
  ROUND(SUM(quantity * unit_price * (1 - discount_pct)), 0) AS revenue_last_7_days
FROM orders
WHERE order_date BETWEEN DATE_SUB('2024-06-05', INTERVAL 7 DAY) AND '2024-06-05';


-- Rolling 30-day churn: customers who were active 30 days ago but no activity in last 30 days
SELECT
  c.customer_id
FROM customers c
LEFT JOIN orders o 
  ON c.customer_id = o.customer_id
  AND o.order_date BETWEEN DATE_SUB(CURDATE(), INTERVAL 30 DAY) AND CURDATE()
WHERE c.last_order_date BETWEEN (DATE_SUB(CURDATE(), INTERVAL 60 DAY)) AND (DATE_SUB(CURDATE(), INTERVAL 31 DAY))
  AND o.order_date IS NULL;



-- -----------------------------------------------------------------------------
-- Product Funnel Logic: View -> Cart -> Purchase
-- -----------------------------------------------------------------------------

-- Count users who viewed, added to cart, and purchased a product
SELECT
  product_id,
  COUNT(DISTINCT CASE WHEN event_type = 'view' THEN user_id END) AS product_views,
  COUNT(DISTINCT CASE WHEN event_type = 'add_to_cart' THEN user_id END) AS product_cart_adds,
  COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END) AS product_purchases
FROM product_events
GROUP BY product_id;



-- -----------------------------------------------------------------------------
-- Conversion Rates by Traffic Source
-- -----------------------------------------------------------------------------

-- Conversion rate = signups / users per source

SELECT
  ts.source,
  COUNT(DISTINCT ts.user_id) AS users,
  COUNT(DISTINCT c.user_id) AS signups,
  ROUND(COUNT(DISTINCT c.user_id) / COUNT(DISTINCT ts.user_id), 1) AS conversion_rate
FROM traffic_sources ts
LEFT JOIN conversions c ON ts.user_id = c.user_id
  AND c.conversion_type = 'signup'
GROUP BY ts.source;



-- -----------------------------------------------------------------------------
-- Outlier Detection or Performance Monitoring (SLA Breaches, Delays)
-- -----------------------------------------------------------------------------

-- SLA breach example: response_time > 500ms or status_code != 200 in last year
SELECT
  service_name,
  COUNT(*) AS breach_count
FROM performance_logs
WHERE (response_time_ms > 500) OR status_code != 200
  AND log_time BETWEEN DATE_SUB(CURDATE(), INTERVAL 1 YEAR) AND CURDATE()
GROUP BY service_name
ORDER BY service_name;



-- -----------------------------------------------------------------------------
-- Customer Segmentation Queries (Behavior or Spend)
-- -----------------------------------------------------------------------------

-- Segment customers by total spend (low, medium, high)
SELECT
  customer_id,
  CASE 
    WHEN total_spend < 500 THEN 'low'
    WHEN (total_spend >= 500) AND (total_spend < 1500) THEN 'medium'
    ELSE 'high'
  END AS spend_segment
FROM customers
ORDER BY customer_id;


-- -----------------------------------------------------------------------------
-- Sales Forecasting Based on Historical Data
-- -----------------------------------------------------------------------------

-- Monthly revenue totals for past 16 months
SELECT
  DATE_FORMAT(order_date, '%Y-%M') AS 'Year-Month',
  ROUND(SUM(quantity * unit_price * (1 - discount_pct)), 0) AS total_revenue
FROM orders
WHERE order_date BETWEEN DATE_SUB(CURDATE(), INTERVAL 16 MONTH) AND CURDATE()
GROUP BY DATE_FORMAT(order_date, '%Y-%M')
ORDER BY 'Year-Month';

-- Calculate month-over-month growth
WITH monthly_revenue AS (
  SELECT
    DATE_FORMAT(order_date, '%Y-%M') AS my_year_month,
    ROUND(SUM(quantity * unit_price * (1 - discount_pct)), 0) AS total_revenue
  FROM orders
  GROUP BY DATE_FORMAT(order_date, '%Y-%M')
)
SELECT 
  my_year_month,
  total_revenue,
  total_revenue / LAG(total_revenue) OVER (ORDER BY my_year_month) - 1 AS monthly_growth
FROM monthly_revenue;

WITH monthly_revenue AS (
  SELECT
    DATE_FORMAT(order_date, '%Y-%M') AS my_year_month,
    ROUND(SUM(quantity * unit_price * (1 - discount_pct)), 0) AS total_revenue
  FROM orders
  GROUP BY DATE_FORMAT(order_date, '%Y-%M')
)
SELECT 
   my_year_month,
  total_revenue,
  ROUND(
    (total_revenue - LAG(total_revenue) OVER (ORDER BY  my_year_month)) 
    / LAG(total_revenue) OVER (ORDER BY  my_year_month), 4
  ) AS monthly_growth
FROM monthly_revenue;


-- -----------------------------------------------------------------------------
-- Repeat Purchase Behavior 
-- -----------------------------------------------------------------------------

-- How many customers made more than 1 purchase
SELECT
  customer_id,
  COUNT(DISTINCT order_id) AS total_orders
FROM orders
GROUP BY customer_id
HAVING total_orders > 1
ORDER BY customer_id;

-- Show percentages of repeat customers
SELECT
  COUNT(DISTINCT CASE WHEN order_count > 1 THEN customer_id END) AS repeat_customers,
  COUNT(DISTINCT customer_id) AS total_customers,
  ROUND(
    COUNT(DISTINCT CASE WHEN order_count > 1 THEN customer_id END) / COUNT(DISTINCT customer_id) * 100,
    1
  ) AS repeat_rate
FROM (
  SELECT
    customer_id,
    COUNT(*) AS order_count
  FROM orders
  GROUP BY customer_id
) AS T1;
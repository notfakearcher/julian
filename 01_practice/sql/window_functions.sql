-- -----------------------------------------------------------------------------
-- ROW_NUMBER(), RANK(), DENSE_RANK()
-- -----------------------------------------------------------------------------

SELECT * 
FROM customers;

SELECT * 
FROM orders;

SELECT *
FROM products;

-- ROW_NUMBER() — First order per customer
SELECT *
FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date ASC) AS row_num
  FROM orders
) AS ranks
WHERE row_num = 1;

-- RANK() / DENSE_RANK() — Top-selling products per category
-- USE RANK() if you want gaps when there are ties, e.g. 1, 1, 3
-- USE DENSE_RANK() if you do not want gaps when there are ties, e.g. 1, 1, 2
SELECT
  p.product_name,
  p.category,
  SUM(o.quantity) AS order_total,
  RANK() OVER (PARTITION BY p.category ORDER BY SUM(o.quantity) DESC) AS order_rank
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY p.category, p.product_name
ORDER BY p.category, order_rank;

SELECT
  p.product_name,
  p.category,
  SUM(o.quantity) AS order_total,
  DENSE_RANK() OVER (PARTITION BY p.category ORDER BY SUM(o.quantity) DESC) AS order_rank
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY p.category, p.product_name
ORDER BY p.category, order_rank;


-- -----------------------------------------------------------------------------
-- Running totals / row-based group averages with PARTITION BY and ORDER BY
-- -----------------------------------------------------------------------------

-- SUM(...) OVER() — Running total of orders per customer
SELECT
  customer_id,
  order_date,
  quantity,
  SUM(quantity) OVER (PARTITION BY customer_id ORDER BY order_date ASC) AS running_total
FROM orders
ORDER BY customer_id, order_date;

-- AVG(...) OVER() —  Average quantity across all orders in the same region, and repeats it on each row
SELECT 
  o.customer_id,
  c.region,
  o.order_id,
  AVG(o.quantity) OVER (PARTITION BY c.region) AS avg_region_quantity
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;


-- -----------------------------------------------------------------------------
-- LEAD() / LAG()
-- -----------------------------------------------------------------------------

-- LAG() / LEAD() — Previous and next order per customer
SELECT
  customer_id,
  order_id,
  order_date,
  LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS previous_order,
  LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order
FROM orders;

-- Time difference between consecutive orders for each customer
SELECT 
  *,
  AVG(days_to_next_order) OVER (PARTITION BY customer_id) AS avg_days_between_orders_by_customer
FROM (
  SELECT
    customer_id,
    order_id,
    order_date,
    LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS previous_order,
    LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order,
    DATEDIFF(
      LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date), 
      order_date
    ) AS days_to_next_order
  FROM orders
) AS T1

-- -----------------------------------------------------------------------------
-- FIRST_VALUE(), LAST_VALUE()
-- -----------------------------------------------------------------------------

-- FIRST_VALUE() — First product ordered per customer
SELECT 
  customer_id,
  order_date,
  product_id,
  FIRST_VALUE(product_id) OVER (PARTITION BY customer_id ORDER BY order_date) AS first_product_ordered
FROM orders
ORDER BY customer_id, order_date;

-- LAST_VALUE() — Last product ordered per customer
SELECT 
  customer_id,
  order_date,
  product_id,
  LAST_VALUE(product_id) OVER (
    PARTITION BY customer_id 
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS last_product_ordered
FROM orders
ORDER BY customer_id, order_date;


-- -----------------------------------------------------------------------------
-- NTILE() (bucketing into quantiles)
-- -----------------------------------------------------------------------------

-- NTILE() — Split customers into quartiles by total order volume
-- Higer volume gets lower rank hence the need for DESC

SELECT
  *,
  NTILE(4) OVER (ORDER BY total_quantity DESC) AS order_volume_quantile
FROM (
  SELECT
    customer_id,
    SUM(quantity) AS total_quantity
  FROM orders
  GROUP BY customer_id
) AS T1;


-- -----------------------------------------------------------------------------
-- Percent-of-total within partition
-- -----------------------------------------------------------------------------

-- Percent of Total Order Quantity per Customer (by Region)
-- What percentage of each region’s total orders was placed by each customer?
SELECT
  *,
  ROUND(
    (cust_region_orders / SUM(cust_region_orders) OVER (PARTITION BY region)) * 100,
    2
  ) AS percent_cust_region_orders
FROM (
  SELECT 
    c.region,
    o.customer_id,
    SUM(o.quantity) AS cust_region_orders
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  GROUP BY c.region, o.customer_id
) AS T1;

SELECT 
  c.region,
  o.customer_id,
  SUM(o.quantity) AS cust_region_orders,
  ROUND(
    (SUM(o.quantity) / SUM(SUM(o.quantity)) OVER (PARTITION BY c.region)) * 100,
    2
  ) AS percent_cust_region_orders
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.region, o.customer_id;


-- Percent of Total Revenue per Product (Category Partition)
-- What percentage of category revenue each product generated
SELECT
  p.category,
  o.product_id,
  SUM(o.quantity * p.price) AS product_revenue,
  ROUND(
    (SUM(o.quantity * p.price) / SUM(SUM(o.quantity * p.price)) OVER (PARTITION BY p.category)) * 100,
    2
  ) AS percent_of_category_revenue
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY p.category, o.product_id;


-- -----------------------------------------------------------------------------
-- Difference from previous row, cumulative sums, rolling sums and averages
-- -----------------------------------------------------------------------------

-- Difference from Previous Row (LAG())
SELECT
  customer_id,
  order_date,
  quantity,
  quantity - LAG(quantity) OVER (PARTITION BY customer_id ORDER BY order_date) AS diff_from_previous_quantity
FROM orders;


-- Rolling 3-Day Quantity Total by Customer (Moving Window)
SELECT
  customer_id,
  order_date,
  quantity,
  SUM(quantity) OVER(
    PARTITION BY customer_id
    ORDER BY order_date
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
  ) AS rolling_3day_total_quantity
FROM orders;


-- Rolling 3-Day Quantity Average by Customer (Moving Window)
SELECT
  customer_id,
  order_date,
  quantity,
  ROUND(
    AVG(quantity) OVER(
      PARTITION BY customer_id
      ORDER BY order_date
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ),
    2
  ) AS rolling_3day_avg_quantity
FROM orders;

-- Cumulative Averages and Cumulative Sums similar but rows unbounded
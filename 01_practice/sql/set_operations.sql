-- -----------------------------------------------------------------------------
-- UNION — Removes Duplicates
-- -----------------------------------------------------------------------------

SELECT customer_name
FROM customers
UNION
SELECT 'Alice' AS customer_name;

-- -----------------------------------------------------------------------------
-- UNION ALL — Keeps All Rows
-- -----------------------------------------------------------------------------

SELECT customer_name
FROM customers
UNION ALL
SELECT 'Alice' AS customer_name;


-- -----------------------------------------------------------------------------
-- Using DISTINCT with UNION ALL
-- -----------------------------------------------------------------------------

SELECT DISTINCT customer_name
FROM (
  SELECT customer_name
  FROM customers
  UNION ALL
  SELECT 'Alice' AS customer_name
) AS T1;
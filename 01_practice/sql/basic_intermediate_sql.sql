-- -----------------------------------------------------------------------------
-- SELECT, FROM, WHERE, ORDER BY
-- -----------------------------------------------------------------------------

-- Retrieving sigle column 
SELECT prod_name
FROM products;

-- Retrieving multiple columns
SELECT prod_id, prod_name, prod_price
FROM products;

-- Retrieving all columns
SELECT *
FROM products;

-- Sorting by single column
SELECT prod_name
FROM products
ORDER BY prod_name;

-- Sorting by multiple columns
SELECT prod_id, prod_price, prod_name
FROM products;

SELECT prod_id, prod_price, prod_name
FROM products
ORDER BY prod_price, prod_name;

-- Sorting by column position
SELECT prod_id, prod_price, prod_name
FROM products
ORDER By 2, 3

-- Sorting by direction
SELECT prod_name
FROM products
ORDER BY prod_name DESC;

SELECT prod_id, prod_price, prod_name
FROM products
ORDER BY prod_price DESC;

SELECT prod_id, prod_price, prod_name
FROM products
ORDER BY prod_price DESC, prod_name;

-- -----------------------------------------------------------------------------
-- LOGICAL OOPERATORS
-- -----------------------------------------------------------------------------

-- Using the WHERE Clause
SELECT prod_name, prod_price
FROM products
WHERE prod_price = 3.49;

SELECT prod_name, prod_price
FROM products
WHERE prod_price <> 3.49;

SELECT prod_name, prod_price
FROM products
WHERE prod_price != 3.49;

SELECT prod_name, prod_price
FROM products
WHERE prod_price < 3.49;

SELECT prod_name, prod_price
FROM products
WHERE prod_price <= 3.49;

SELECT prod_name, prod_price
FROM products
WHERE prod_price > 3.49;

SELECT prod_name, prod_price
FROM products
WHERE prod_price >= 3.49;

SELECT prod_name, prod_price
FROM products
WHERE prod_price BETWEEN 4 AND 10;

SELECT prod_name, prod_price
FROM products
WHERE prod_price is NULL;

SELECT prod_name, prod_price
FROM products
WHERE prod_price is NOT NULL;

-- Using the AND operator
SELECT prod_id, prod_price, prod_name
FROM products
WHERE vend_id = 'DLL01' AND prod_price <=4;

-- Using the OR operator
SELECT prod_name, prod_price
FROM products
WHERE vend_id = 'DL101' OR vend_id = 'BRS01';

SELECT prod_name, prod_price, vend_id
FROM products
WHERE vend_id = 'DLL01' OR vend_id = 'BRS01'
ORDER BY prod_name;

-- Understanding Order of Evaluation
SELECT prod_name, prod_price
FROM products
WHERE vend_id = 'DLL01' OR vend_id = 'BRS01' AND prod_price >= 10;

SELECT prod_name, prod_price
FROM products
WHERE (vend_id = 'DLL01' OR vend_id = 'BRS01') AND (prod_price >= 10);

-- Using the IN Operator
SELECT prod_name, prod_price, vend_id
FROM products
WHERE vend_id IN ('DLL01', 'BRS01')
ORDER BY prod_name;

-- Using the NOT Operator
SELECT prod_name, vend_id
FROM products
WHERE NOT vend_id = 'DLL01';

-- The percent sign (%) wildcard
SELECT prod_id, prod_name
FROM products
WHERE prod_name LIKE 'Fish%';

SELECT prod_id, prod_name
FROM products
WHERE prod_name LIKE '%bean bag%';

SELECT prod_name
FROM products
WHERE prod_name LIKE 'F%y';

SELECT prod_id, prod_name
FROM products
WHERE prod_name LIKE '% inch teddy bear';

-- The underscore (_) wildcard
SELECT prod_id, prod_name
FROM products
WHERE prod_name LIKE '_ inch teddy bear';

SELECT prod_id, prod_name
FROM products
WHERE prod_name LIKE '__ inch teddy bear';

-- The brackets ([ ]) wildcard equivalent is REGEXP
SELECT cust_contact
FROM customers
WHERE cust_contact REGEXP '^[JM]';

SELECT cust_contact
FROM customers
WHERE cust_contact NOT REGEXP '^[JM]';

SELECT cust_contact
FROM customers
WHERE cust_contact REGEXP '[sn]$';


-- -----------------------------------------------------------------------------
-- DISTINCT, LIMIT / TOP, OFFSET
-- -----------------------------------------------------------------------------

-- Retrieving distinct rows
SELECT vend_id
FROM products;

SELECT DISTINCT vend_id
FROM products;

-- Limiting results
SELECT prod_name
FROM products;

SELECT prod_name
FROM products
LIMIT 5;

SELECT prod_name
FROM products
LIMIT 5 OFFSET 5;


-- -----------------------------------------------------------------------------
-- SORTING AND NULL ORDERING
-- -----------------------------------------------------------------------------
SELECT *
FROM customers;

-- Basic sorting (NULLs first in ASC)
SELECT cust_email
FROM customers
ORDER BY cust_email ASC;


-- Basic sorting (NULLs last in DESC)
SELECT cust_email
FROM customers
ORDER BY cust_email DESC;

-- Reverse sorting (NULLs last in ASC)
SELECT cust_email
FROM customers
ORDER BY cust_email IS NULL, cust_email ASC;

-- Reverse sorting (NULLs first in DESC)
SELECT cust_email
FROM customers
ORDER BY cust_email IS NOT NULL, cust_email DESC;


-- -----------------------------------------------------------------------------
-- BASIC AGGREGATIONS: COUNT, SUM, AVG, MIN, MAX
-- -----------------------------------------------------------------------------

-- The AVG() function

SELECT prod_name, prod_price
FROM products;

SELECT AVG(prod_price) AS avg_price
FROM products;

SELECT vend_id, AVG(prod_price) AS avg_price
FROM products
WHERE vend_id = 'DLL01';

SELECT vend_id, prod_price
FROM products;

-- The COUNT() function
SELECT *
FROM customers;

SELECT COUNT(*) AS num_cust
FROM customers;

SELECT COUNT(DISTINCT cust_name) AS num_cust
FROM customers;

-- The MAX function
SELECT prod_name, prod_price
FROM products;

SELECT MAX(prod_price) as max_price
FROM products;

-- The MIN function
SELECT prod_name, prod_price
FROM products;

SELECT MIN(prod_price) as min_price
FROM products;

-- The SUM function 
SELECT SUM(quantity) AS items_ordered
FROM orderitems
WHERE order_num = 20005;

SELECT SUM(item_price * quantity) AS total_price
FROM orderitems
WHERE order_num = 20005;

-- Aggregates on Distinct Values
SELECT AVG(DISTINCT prod_price) AS avg_price
FROM products
WHERE vend_id = 'DLL01'

-- Combining aggregate functions
SELECT 
  COUNT(*) AS num_items,
  MIN(prod_price) AS price_min,
  MAX(prod_price) AS price_max,
  AVG(prod_price) AS price_avg,
  SUM(prod_price) AS price_total
FROM products;


-- -----------------------------------------------------------------------------
-- GROUP BY + HAVING
-- -----------------------------------------------------------------------------

-- Creating Groups
SELECT vend_id, COUNT(*) AS num_prods
FROM products
GROUP BY vend_id;

-- Filtering Groups
SELECT cust_id, COUNT(*) AS orders
FROM orders 
GROUP by cust_id
HAVING COUNT(*) >= 2;

SELECT vend_id, COUNT(*) AS num_prods
FROM products
WHERE prod_price >= 4
GROUP BY vend_id
HAVING COUNT(*) >= 2;

SELECT order_num, COUNT(*) AS items
FROM orderitems
GROUP BY order_num
HAVING COUNT(*) >= 3;

SELECT order_num, COUNT(*) AS items
FROM orderitems
GROUP BY order_num
HAVING COUNT(*) >= 3
ORDER BY items, order_num;


-- -----------------------------------------------------------------------------
-- CASE WHEN STATEMENTS
-- -----------------------------------------------------------------------------

-- Simple CASE Expression
SELECT *
FROM vendors;

SELECT  *,
        CASE vend_state
          WHEN 'CA' THEN 'California'
          WHEN 'MI' THEN 'Michigan'
          WHEN 'NY' THEN 'New York'
          WHEN 'OH' THEN 'Ohio'
          ELSE 'Unknown'
        END AS vend_state_full
FROM vendors
ORDER BY vend_state_full;

-- Searched CASE Expression

SELECT * 
FROM orderitems;

SELECT  *,
        CASE 
          WHEN quantity >= 50 THEN 'large'
          WHEN quantity >= 20 THEN 'medium'
          WHEN quantity >= 10 THEN 'small'
          ELSE 'very small'
        END AS order_size_desc
FROM orderitems;

SELECT  *,
        CASE 
          WHEN quantity >= 50 THEN 'large'
          WHEN quantity BETWEEN 20 AND 50 THEN 'medium'
          WHEN quantity >= 10 THEN 'small'
          ELSE 'very small'
        END AS order_size_desc
FROM orderitems;

-- CASE in ORDER BY
SELECT  *
FROM vendors
ORDER BY CASE vend_state
          WHEN 'MI' THEN 1
          WHEN 'CA' THEN 2
          WHEN 'NY' THEN 3
          WHEN 'OH' THEN 4
          ELSE 0
        END;

-- CASE with Aggregate Functions
SELECT *
FROM orders;

SELECT  COUNT(*) AS total_orders,
        COUNT(CASE WHEN order_date BETWEEN '2020-01-01' AND '2020-01-31' THEN 1 END) AS 'jan_orders'
FROM orders;

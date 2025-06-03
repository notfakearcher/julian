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
-- SELECT COLUMN
-- FROM customers;
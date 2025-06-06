-- -----------------------------------------------------------------------------
-- HANDLING NULLS
-- -----------------------------------------------------------------------------

-- Using IS NULL
SELECT *
FROM vendors;

SELECT *
FROM vendors
WHERE vend_state IS NULL;

SELECT *
FROM vendors
WHERE NOT vend_state IS NULL;

-- Using COASLESCE
SELECT *
FROM customers;

SELECT cust_name, COALESCE(cust_email, 'No Email') AS cust_email
FROM customers;

SELECT cust_name, COALESCE(cust_email, 0) AS cust_email
FROM customers;

SELECT * 
FROM products;

-- Using IFNULL
SELECT cust_name, IFNULL(cust_email, 'No Email') AS cust_email
FROM customers;
SELECT cust_name, IFNULL(cust_email, 0) AS cust_email
FROM customers;



-- -----------------------------------------------------------------------------
-- STRING FUNCTIONS
-- -----------------------------------------------------------------------------

-- Using SUBSTRING

SELECT *
FROM vendors;

SELECT 
  vend_id, 
  SUBSTRING(vend_id, 1, 3) AS vend_id_text,
  SUBSTRING(vend_id, 4, 2) AS ven_id_number
FROM vendors;

-- Using LEFT/RIGHT
SELECT
  vend_id,
  LEFT(vend_id, 3) AS vend_id_text,
  RIGHT(vend_id, 2) AS vend_id_number
FROM vendors;

-- Using CONCAT
SELECT *
FROM customers;

SELECT CONCAT(cust_contact, ' <', IFNULL(cust_email, 'No Email'), '>') AS cust_contact_info
FROM customers;

-- Using TRIM (default BOTH LEADING and TRAILING, can specify either one too)
SELECT
  cust_contact,
  HEX(CONCAT('    pre - ', cust_contact, ' - post  '))  AS cust_contact_hex,
  CONCAT('    pre - ', cust_contact, ' - post  ') AS cust_contact_extra_spaces,
  TRIM(
    LEADING ' '
    FROM
    CONCAT('    pre - ', cust_contact, ' - post  ')
  ) AS cust_contact_no_space_leading,
  TRIM(
    TRAILING ' '
    FROM
    CONCAT('    pre - ', cust_contact, ' - post  ')
  ) AS cust_contact_no_space_trailing,
    TRIM(
    TRAILING ' '
    FROM
    CONCAT('    pre - ', cust_contact, ' - post  ')
  ) AS cust_contact_no_space_trailing
FROM customers;

-- Using REPLACE
SELECT 
  cust_contact,
  TRIM(cust_contact) cust_contact2,
  REPLACE(TRIM(cust_contact), '.', '') AS cust_contact3,
  REPLACE(REPLACE(TRIM(cust_contact), '.', ''), ' ', '.' ) AS cust_contact4
FROM customers;


-- -----------------------------------------------------------------------------
-- DATE/TIME FUNCTIONS
-- -----------------------------------------------------------------------------

-- Using NOW(), CURRENT_DATE
SELECT *
FROM orders;

SELECT NOW() AS current_datetime;

SELECT CURRENT_DATE AS today_date;


SELECT
  order_date,
  NOW() AS today_datetime,
  CURRENT_DATE AS today_date
FROM orders;

-- USING DATEDIFF
SELECT
  order_date,
  NOW() AS today_datetime,
  CURRENT_DATE AS today_date,
  DATEDIFF(NOW(), order_date) AS elapsed_days
FROM orders;


-- Using EXTRACT (part FROM date), DAYOFYEAR, DAYOFMONTH, DAYOFWEEK
SELECT
  order_date,
  EXTRACT(YEAR FROM order_date) AS order_year,
  EXTRACT(MONTH FROM order_date) AS order_month,
  EXTRACT(DAY FROM order_date) AS order_day,

  EXTRACT(QUARTER FROM order_date) AS order_quarter,
  EXTRACT(WEEK FROM order_date) AS order_week,

  DAYOFWEEK(order_date) AS order_dayofweek,
  DAYOFMONTH(order_date) AS order_dayofmonth,
  DAYOFYEAR(order_date) AS order_dayofyear,

  NOW() AS today_date,
  EXTRACT(HOUR FROM NOW()) AS current_hour,
  EXTRACT(MINUTE FROM NOW()) AS current_minute,
  EXTRACT(SECOND FROM NOW()) AS current_second,
  EXTRACT(MICROSECOND FROM NOW()) AS current_microsecond
FROM orders;


FROM orders;

-- -----------------------------------------------------------------------------
-- TYPE CASTING/ CONVERSION
-- -----------------------------------------------------------------------------

-- Using CAST(expr AS type): SIGNED/UNSIGNED, CHAR/CHARACTER, BINARY 
-- Using CONVERT is similar; just don't use AS key word and use ',' instead
SELECT *
FROM orderitems;

SELECT CAST('123' AS UNSIGNED) AS as_number;
SELECT CAST('-123' AS SIGNED) AS as_number;

SELECT CAST(45 AS CHAR) AS as_char;

SELECT CAST(12 AS BINARY) AS as_binary;
-- SELECT BIN(12) AS as_binary;


-- Using CAST(expr AS type) :DECIMAL(M,D)
SELECT
  quantity,
  CAST(quantity AS DECIMAL(5,2)) AS quantiy_decimal
FROM orderitems;

SELECT CAST('-123' AS DECIMAL(5,2)) AS as_deciaml;
SELECT CONVERT('-123', DECIMAL(5,2)) AS as_deciaml;





-- Using CAST(expr AS type): DATE, DATETIME, TIME
SELECT CAST('2025-06-01' AS DATE) AS today_date;

SELECT CAST('2025-06-01' AS DATETIME) AS today_datetime;

SELECT CAST('2025-06-01 21:09:37' AS TIME) AS today_time;



-- -----------------------------------------------------------------------------
-- DATA STANDARDIZATION
-- -----------------------------------------------------------------------------

-- Using upper/lower casing
SELECT *
FROM vendors;

SELECT
  vend_name,
  LOWER(CONCAT(REPLACE(REPLACE(TRIM(vend_name), '.', ''), ' ', '.'), '@gmail.com')) AS vend_email_lower,
  UPPER(CONCAT(REPLACE(REPLACE(TRIM(vend_name), '.', ''), ' ', '.'), '@gmail.com')) AS vend_email_upper
FROM vendors;


SELECT 
REGEXP_REPLACE(
  LOWER(CONCAT(REPLACE(REPLACE(TRIM(vend_name), '.', ''), ' ', '.'), '@gmail.com')),
  '\\W+',
  ', '
) AS vend_email_striped
FROM vendors;

-- Using DATE_FORMAT
SELECT
  order_date,
  DATE_FORMAT(order_date, '%Y-%m-%d') AS order_date2,
  DATE_FORMAT(order_date, '%Y-%b-%d') AS order_date3,
  DATE_FORMAT(order_date, '%Y-%M-%d') AS order_date4,

  DATE_FORMAT(order_date, '%y-%b-%d') AS order_date4,
  DATE_FORMAT(order_date, '%Y-%b-%e') AS order_date5,

  DATE_FORMAT(order_date, '%y-%m-%d') AS order_date6,

  DATE_FORMAT(order_date, '%Y-%m-%e') AS order_date7,
  DATE_FORMAT(order_date, '%Y-%m-%D') AS order_date8,

  DATE_FORMAT(order_date, '%a') AS order_dayofweek_txt,

  DATE_FORMAT(order_date, '%Y-01-01') AS year_start,
  DATE_FORMAT(order_date, '%Y-%m-01') AS month_start
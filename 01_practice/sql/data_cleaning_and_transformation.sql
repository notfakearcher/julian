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



-- -----------------------------------------------------------------------------
-- TYPE CASTING/ CONVERSION
-- -----------------------------------------------------------------------------




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
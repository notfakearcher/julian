-- Sell all databases
SHOW DATABASES;

-- See all tables in a specific database
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'upskill';

-- Show all columns in a given database table
SHOW columns
FROM upskill.customers;

SHOW columns
FROM upskill.orderitems;

SHOW columns
FROM upskill.orders;

SHOW columns
FROM upskill.products;

SHOW columns
FROM upskill.vendors;
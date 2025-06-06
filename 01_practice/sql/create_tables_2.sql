-- -----------------------------------------------------------------------------
-- Create Database upskill2
-- -----------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS upskill2;
USE upskill2;


-- -----------------------------------------------------------------------------
-- Create Table customers
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customers (
  customer_id INT PRIMARY KEY,
  customer_name VARCHAR(50),
  region VARCHAR(20)
);

INSERT IGNORE INTO customers (customer_id, customer_name, region) VALUES
(1, 'Alice', 'US'),
(2, 'Bob', 'EU'),
(3, 'Charlie', 'US'),
(4, 'Diana', 'EU'),
(5, 'Ethan', 'IN');



-- -----------------------------------------------------------------------------
-- Create Table products
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
  product_id INT PRIMARY KEY,
  product_name VARCHAR(50),
  category VARCHAR(20),
  price DECIMAL(10,2)
);

INSERT IGNORE INTO products (product_id, product_name, category, price) VALUES
(101, 'Echo Dot', 'Electronics', 49.99),
(102, 'Fire TV', 'Electronics', 39.99),
(103, 'Kindle', 'Books', 89.99),
(104, 'Yoga Mat', 'Sports', 25.00),
(105, 'Protein Powder', 'Health', 29.99);



-- -----------------------------------------------------------------------------
-- Create Table orders
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
  order_id INT PRIMARY KEY,
  customer_id INT,
  product_id INT,
  order_date DATE,
  quantity INT,
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
);

INSERT IGNORE INTO orders (order_id, customer_id, product_id, order_date, quantity) VALUES
(1, 1, 101, '2024-01-01', 1),
(2, 2, 102, '2024-01-02', 2),
(3, 1, 103, '2024-01-03', 1),
(4, 3, 104, '2024-01-04', 3),
(5, 4, 101, '2024-01-05', 1),
(6, 5, 105, '2024-01-06', 2),
(7, 1, 102, '2024-01-07', 1),
(8, 2, 103, '2024-01-08', 2),
(9, 3, 101, '2024-01-09', 1),
(10, 4, 105, '2024-01-10', 3),
(11, 5, 104, '2024-01-11', 1),
(12, 1, 104, '2024-01-12', 2),
(13, 2, 105, '2024-01-13', 1),
(14, 3, 103, '2024-01-14', 1),
(15, 4, 102, '2024-01-15', 2);
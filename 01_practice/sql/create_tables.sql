-- Example table creation scripts for MySQL & MariaDB
-- From "Sams Teach Yourself SQL in 10 Minutes, 5th Edition"
-- http://forta.com/books/0135182794/


-- ----------------------
-- Create Database upskill
-- ----------------------
CREATE DATABASE IF NOT EXISTS upskill;
USE upskill;

-- ----------------------
-- Create Customers table
-- ----------------------
CREATE TABLE Customers (
  cust_id      CHAR(10)  NOT NULL,
  cust_name    CHAR(50)  NOT NULL,
  cust_address CHAR(50),
  cust_city    CHAR(50),
  cust_state   CHAR(5),
  cust_zip     CHAR(10),
  cust_country CHAR(50),
  cust_contact CHAR(50),
  cust_email   CHAR(255),
  PRIMARY KEY (cust_id)
);

-- --------------------
-- Create Vendors table
-- --------------------
CREATE TABLE Vendors (
  vend_id      CHAR(10) NOT NULL,
  vend_name    CHAR(50) NOT NULL,
  vend_address CHAR(50),
  vend_city    CHAR(50),
  vend_state   CHAR(5),
  vend_zip     CHAR(10),
  vend_country CHAR(50),
  PRIMARY KEY (vend_id)
);

-- ---------------------
-- Create Products table
-- ---------------------
CREATE TABLE Products (
  prod_id     CHAR(10)      NOT NULL,
  vend_id     CHAR(10)      NOT NULL,
  prod_name   CHAR(255)     NOT NULL,
  prod_price  DECIMAL(8,2)  NOT NULL,
  prod_desc   TEXT,
  PRIMARY KEY (prod_id),
  FOREIGN KEY (vend_id) REFERENCES Vendors(vend_id)
);


-- -----------------------
-- Create OrderItems table
-- -----------------------
CREATE TABLE OrderItems (
  order_num   INT         NOT NULL,
  order_item  INT         NOT NULL,
  prod_id     CHAR(10)    NOT NULL,
  quantity    INT         NOT NULL,
  item_price  DECIMAL(8,2) NOT NULL,
  PRIMARY KEY (order_num, order_item),
  FOREIGN KEY (order_num) REFERENCES Orders(order_num),
  FOREIGN KEY (prod_id) REFERENCES Products(prod_id)
);





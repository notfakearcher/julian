-- -----------------------------------------------------------------------------
-- Create Database amazon
-- -----------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS amazon;
USE amazon;


-- -----------------------------------------------------------------------------
-- Create orders — Revenue by Category/Customer/Product
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS orders (
  order_id INT PRIMARY KEY,
  customer_id INT,
  product_id INT,
  order_date DATE,
  quantity INT,
  unit_price DECIMAL(10,2),
  discount_pct DECIMAL(4,2)
);

INSERT IGNORE INTO orders (order_id, customer_id, product_id, order_date, quantity, unit_price, discount_pct) VALUES
(1, 108, 7, '2024-03-15', 4, 137.66, 0.10),
(2, 115, 5, '2024-03-08', 3, 118.88, 0.10),
(3, 104, 10, '2024-06-13', 4, 168.41, 0.00),
(4, 116, 9, '2024-03-11', 2, 111.76, 0.00),
(5, 113, 4, '2024-03-02', 2, 165.55, 0.00),
(6, 116, 2, '2024-06-03', 2, 135.25, 0.05),
(7, 104, 6, '2024-03-21', 2, 17.04, 0.15),
(8, 106, 2, '2024-01-09', 3, 154.87, 0.10),
(9, 102, 6, '2024-06-16', 3, 135.89, 0.15),
(10, 114, 6, '2024-05-26', 3, 179.28, 0.15),
(11, 113, 6, '2024-05-10', 1, 104.11, 0.10),
(12, 104, 10, '2024-03-13', 5, 30.88, 0.05),
(13, 105, 5, '2024-02-17', 4, 179.99, 0.10),
(14, 105, 9, '2024-04-09', 3, 171.31, 0.10),
(15, 114, 4, '2024-03-30', 4, 120.55, 0.00),
(16, 116, 7, '2024-03-26', 2, 86.84, 0.05),
(17, 117, 7, '2024-02-16', 1, 95.69, 0.15),
(18, 104, 9, '2024-05-02', 5, 189.84, 0.00),
(19, 106, 3, '2024-04-03', 4, 161.84, 0.05),
(20, 120, 2, '2024-03-15', 1, 179.40, 0.00),
(21, 118, 3, '2024-02-14', 4, 131.90, 0.05),
(22, 106, 10, '2024-06-20', 4, 90.56, 0.10),
(23, 102, 2, '2024-06-29', 1, 57.14, 0.00),
(24, 102, 4, '2024-05-14', 3, 42.33, 0.15),
(25, 116, 5, '2024-01-25', 4, 131.45, 0.15),
(26, 109, 6, '2024-02-08', 1, 104.46, 0.05),
(27, 115, 2, '2024-01-22', 5, 61.72, 0.15),
(28, 113, 2, '2024-03-17', 2, 162.01, 0.15),
(29, 115, 2, '2024-05-30', 4, 46.01, 0.00),
(30, 114, 4, '2024-03-19', 5, 115.76, 0.00),
(31, 103, 3, '2024-04-14', 4, 40.10, 0.15),
(32, 101, 5, '2024-03-27', 1, 177.91, 0.10),
(33, 107, 5, '2024-04-10', 1, 54.02, 0.05),
(34, 115, 6, '2024-03-03', 3, 91.63, 0.00),
(35, 117, 3, '2024-02-21', 4, 114.47, 0.10),
(36, 113, 4, '2024-06-09', 3, 37.37, 0.15),
(37, 114, 4, '2024-05-30', 2, 133.31, 0.00),
(38, 101, 2, '2024-03-26', 2, 150.34, 0.00),
(39, 113, 6, '2024-03-01', 4, 133.55, 0.15),
(40, 115, 1, '2024-04-26', 1, 101.14, 0.10),
(41, 104, 9, '2024-04-03', 3, 174.57, 0.15),
(42, 114, 10, '2024-06-15', 2, 157.12, 0.10),
(43, 104, 2, '2024-01-11', 3, 97.76, 0.00),
(44, 101, 10, '2024-02-27', 1, 136.82, 0.00),
(45, 104, 1, '2024-03-26', 4, 163.58, 0.15),
(46, 101, 4, '2024-06-25', 1, 82.79, 0.10),
(47, 103, 2, '2024-04-04', 4, 95.27, 0.10),
(48, 118, 10, '2024-05-22', 2, 190.26, 0.05),
(49, 113, 2, '2024-01-18', 2, 12.80, 0.05),
(50, 104, 6, '2024-01-19', 3, 100.97, 0.00),
(51, 108, 7, '2024-01-17', 2, 190.87, 0.05),
(52, 118, 6, '2024-04-28', 5, 74.69, 0.05),
(53, 116, 2, '2024-02-22', 2, 162.88, 0.00),
(54, 117, 9, '2024-03-18', 1, 177.47, 0.05),
(55, 118, 10, '2024-04-12', 3, 166.78, 0.10),
(56, 119, 3, '2024-03-31', 2, 156.36, 0.10),
(57, 118, 5, '2024-01-09', 3, 148.79, 0.15),
(58, 120, 6, '2024-06-11', 2, 60.68, 0.05),
(59, 108, 1, '2024-06-06', 3, 125.14, 0.00),
(60, 117, 3, '2024-06-20', 2, 135.28, 0.05),
(61, 111, 2, '2024-02-11', 2, 121.59, 0.05),
(62, 119, 9, '2024-02-16', 4, 187.66, 0.05),
(63, 117, 8, '2024-04-18', 1, 186.45, 0.10),
(64, 104, 9, '2024-06-20', 5, 154.22, 0.05),
(65, 110, 2, '2024-01-31', 4, 57.99, 0.05),
(66, 109, 7, '2024-01-15', 4, 165.07, 0.10),
(67, 119, 1, '2024-05-07', 2, 23.50, 0.05),
(68, 107, 5, '2024-06-20', 1, 23.25, 0.15),
(69, 104, 9, '2024-04-26', 4, 99.40, 0.05),
(70, 109, 1, '2024-03-15', 2, 34.34, 0.05),
(71, 101, 3, '2024-01-09', 3, 180.32, 0.00),
(72, 103, 9, '2024-03-12', 2, 115.53, 0.05),
(73, 120, 10, '2024-01-16', 5, 103.49, 0.10),
(74, 115, 6, '2024-06-21', 4, 20.10, 0.15),
(75, 117, 4, '2024-01-13', 1, 58.69, 0.00),
(76, 120, 10, '2024-01-12', 1, 182.25, 0.00),
(77, 118, 8, '2024-03-06', 2, 38.71, 0.10),
(78, 107, 10, '2024-04-11', 1, 26.62, 0.10),
(79, 104, 7, '2024-03-07', 5, 149.86, 0.05),
(80, 101, 9, '2024-01-29', 5, 66.79, 0.15),
(81, 106, 1, '2024-03-01', 2, 178.27, 0.05),
(82, 111, 10, '2024-04-24', 2, 125.83, 0.05),
(83, 102, 2, '2024-04-19', 1, 109.84, 0.15),
(84, 115, 1, '2024-05-25', 3, 22.94, 0.05),
(85, 102, 5, '2024-03-25', 4, 108.59, 0.00),
(86, 103, 1, '2024-03-04', 2, 69.89, 0.15),
(87, 118, 4, '2024-06-26', 2, 71.26, 0.10),
(88, 119, 4, '2024-05-27', 1, 92.00, 0.10),
(89, 113, 9, '2024-05-24', 5, 26.87, 0.15),
(90, 102, 6, '2024-06-07', 4, 12.29, 0.15),
(91, 117, 1, '2024-06-08', 5, 140.69, 0.00),
(92, 116, 10, '2024-02-18', 4, 103.50, 0.05),
(93, 114, 1, '2024-03-15', 1, 189.84, 0.10),
(94, 104, 9, '2024-04-30', 3, 101.75, 0.10),
(95, 109, 3, '2024-03-05', 4, 199.27, 0.10),
(96, 111, 5, '2024-06-09', 1, 123.36, 0.00),
(97, 113, 6, '2024-02-15', 3, 108.65, 0.10),
(98, 116, 3, '2024-05-23', 3, 106.82, 0.15),
(99, 116, 4, '2024-05-08', 2, 121.45, 0.00),
(100, 105, 6, '2024-04-09', 2, 44.61, 0.00);





-- -----------------------------------------------------------------------------
-- Create products Table Definition
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS products (
  product_id INT PRIMARY KEY,
  product_name VARCHAR(100),
  category VARCHAR(50),
  base_price DECIMAL(10,2)
);

INSERT INTO products (product_id, product_name, category, base_price) VALUES
(1, '4K LED TV', 'Electronics', 129.99),
(2, 'Wireless Headphones', 'Electronics', 149.99),
(3, 'Espresso Machine', 'Home Appliances', 159.99),
(4, 'Blender Pro 500', 'Kitchen', 110.00),
(5, 'Smartphone X2', 'Mobile', 170.00),
(6, 'Fitness Tracker', 'Wearables', 99.99),
(7, 'Running Shoes Z', 'Footwear', 145.00),
(8, 'Winter Jacket', 'Apparel', 175.00),
(9, 'Office Chair', 'Furniture', 120.00),
(10, 'Tablet Mini 10', 'Electronics', 140.00);


-- -----------------------------------------------------------------------------
-- Create user_activity — DAU, MAU, Retention
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS user_activity (
  user_id INT,
  activity_date DATE,
  activity_type ENUM('login', 'search', 'purchase', 'logout'),
  session_id VARCHAR(50)
);


INSERT IGNORE INTO user_activity (user_id, activity_date, activity_type, session_id) VALUES
(101, '2024-01-08', 'login', 'session_101_1'),
(101, '2024-01-08', 'search', 'session_101_1'),
(101, '2024-01-08', 'logout', 'session_101_1'),

(101, '2024-03-25', 'login', 'session_101_2'),
(101, '2024-03-25', 'purchase', 'session_101_2'),
(101, '2024-03-25', 'logout', 'session_101_2'),

(101, '2024-06-24', 'login', 'session_101_3'),
(101, '2024-06-24', 'search', 'session_101_3'),
(101, '2024-06-24', 'logout', 'session_101_3'),
(102, '2024-04-18', 'login', 'session_102_1'),
(102, '2024-04-18', 'search', 'session_102_1'),
(102, '2024-04-18', 'logout', 'session_102_1'),

(102, '2024-05-13', 'login', 'session_102_2'),
(102, '2024-05-13', 'add_to_cart', 'session_102_2'),
(102, '2024-05-13', 'logout', 'session_102_2'),

(102, '2024-06-06', 'login', 'session_102_3'),
(102, '2024-06-06', 'purchase', 'session_102_3'),
(102, '2024-06-06', 'logout', 'session_102_3'),
(103, '2024-03-11', 'login', 'session_103_1'),
(103, '2024-03-11', 'add_to_cart', 'session_103_1'),
(103, '2024-03-11', 'logout', 'session_103_1'),

(103, '2024-04-03', 'login', 'session_103_2'),
(103, '2024-04-03', 'search', 'session_103_2'),
(103, '2024-04-03', 'logout', 'session_103_2'),

(103, '2024-04-13', 'login', 'session_103_3'),
(103, '2024-04-13', 'search', 'session_103_3'),
(103, '2024-04-13', 'logout', 'session_103_3'),
(104, '2024-01-10', 'login', 'session_104_1'),
(104, '2024-01-10', 'add_to_cart', 'session_104_1'),
(104, '2024-01-10', 'logout', 'session_104_1'),

(104, '2024-03-06', 'login', 'session_104_2'),
(104, '2024-03-06', 'search', 'session_104_2'),
(104, '2024-03-06', 'logout', 'session_104_2'),

(104, '2024-03-12', 'login', 'session_104_3'),
(104, '2024-03-12', 'add_to_cart', 'session_104_3'),
(104, '2024-03-12', 'logout', 'session_104_3'),

(104, '2024-03-20', 'login', 'session_104_4'),
(104, '2024-03-20', 'purchase', 'session_104_4'),
(104, '2024-03-20', 'logout', 'session_104_4'),

(104, '2024-04-02', 'login', 'session_104_5'),
(104, '2024-04-02', 'search', 'session_104_5'),
(104, '2024-04-02', 'logout', 'session_104_5'),

(104, '2024-05-01', 'login', 'session_104_6'),
(104, '2024-05-01', 'search', 'session_104_6'),
(104, '2024-05-01', 'logout', 'session_104_6'),

(104, '2024-06-12', 'login', 'session_104_7'),
(104, '2024-06-12', 'search', 'session_104_7'),
(104, '2024-06-12', 'logout', 'session_104_7'),
(105, '2024-02-16', 'login', 'session_105_1'),
(105, '2024-02-16', 'search', 'session_105_1'),
(105, '2024-02-16', 'logout', 'session_105_1'),

(105, '2024-04-08', 'login', 'session_105_2'),
(105, '2024-04-08', 'search', 'session_105_2'),
(105, '2024-04-08', 'logout', 'session_105_2'),

(106, '2024-01-08', 'login', 'session_106_1'),
(106, '2024-01-08', 'search', 'session_106_1'),
(106, '2024-01-08', 'logout', 'session_106_1'),

(106, '2024-04-02', 'login', 'session_106_2'),
(106, '2024-04-02', 'search', 'session_106_2'),
(106, '2024-04-02', 'logout', 'session_106_2'),

(106, '2024-06-19', 'login', 'session_106_3'),
(106, '2024-06-19', 'purchase', 'session_106_3'),
(106, '2024-06-19', 'logout', 'session_106_3'),

(107, '2024-04-09', 'login', 'session_107_1'),
(107, '2024-04-09', 'search', 'session_107_1'),
(107, '2024-04-09', 'logout', 'session_107_1'),

(107, '2024-06-19', 'login', 'session_107_2'),
(107, '2024-06-19', 'purchase', 'session_107_2'),
(107, '2024-06-19', 'logout', 'session_107_2'),

(108, '2024-01-16', 'login', 'session_108_1'),
(108, '2024-01-16', 'search', 'session_108_1'),
(108, '2024-01-16', 'logout', 'session_108_1'),

(108, '2024-03-14', 'login', 'session_108_2'),
(108, '2024-03-14', 'search', 'session_108_2'),
(108, '2024-03-14', 'logout', 'session_108_2'),

(108, '2024-06-05', 'login', 'session_108_3'),
(108, '2024-06-05', 'search', 'session_108_3'),
(108, '2024-06-05', 'logout', 'session_108_3'),

(109, '2024-01-14', 'login', 'session_109_1'),
(109, '2024-01-14', 'search', 'session_109_1'),
(109, '2024-01-14', 'logout', 'session_109_1'),

(109, '2024-03-14', 'login', 'session_109_2'),
(109, '2024-03-14', 'purchase', 'session_109_2'),

(109, '2024-03-14', 'logout', 'session_109_2'),
(110, '2024-01-30', 'login', 'session_110_1'),
(110, '2024-01-30', 'search', 'session_110_1'),
(110, '2024-01-30', 'logout', 'session_110_1'),

(111, '2024-02-10', 'login', 'session_111_1'),
(111, '2024-02-10', 'search', 'session_111_1'),
(111, '2024-02-10', 'logout', 'session_111_1'),

(111, '2024-04-23', 'login', 'session_111_2'),
(111, '2024-04-23', 'search', 'session_111_2'),
(111, '2024-04-23', 'logout', 'session_111_2'),

(111, '2024-06-08', 'login', 'session_111_3'),
(111, '2024-06-08', 'search', 'session_111_3'),
(111, '2024-06-08', 'logout', 'session_111_3'),

(113, '2024-01-17', 'login', 'session_113_1'),
(113, '2024-01-17', 'search', 'session_113_1'),
(113, '2024-01-17', 'logout', 'session_113_1'),

(113, '2024-02-14', 'login', 'session_113_2'),
(113, '2024-02-14', 'purchase', 'session_113_2'),
(113, '2024-02-14', 'logout', 'session_113_2'),

(113, '2024-03-01', 'login', 'session_113_3'),
(113, '2024-03-01', 'search', 'session_113_3'),
(113, '2024-03-01', 'logout', 'session_113_3'),

(113, '2024-03-16', 'login', 'session_113_4'),
(113, '2024-03-16', 'add_to_cart', 'session_113_4'),
(113, '2024-03-16', 'logout', 'session_113_4'),

(113, '2024-05-09', 'login', 'session_113_5'),
(113, '2024-05-09', 'search', 'session_113_5'),
(113, '2024-05-09', 'logout', 'session_113_5'),

(113, '2024-05-23', 'login', 'session_113_6'),
(113, '2024-05-23', 'search', 'session_113_6'),
(113, '2024-05-23', 'logout', 'session_113_6'),

(113, '2024-06-08', 'login', 'session_113_7'),
(113, '2024-06-08', 'add_to_cart', 'session_113_7'),
(113, '2024-06-08', 'logout', 'session_113_7'),

(114, '2024-03-14', 'login', 'session_114_1'),
(114, '2024-03-14', 'purchase', 'session_114_1'),
(114, '2024-03-14', 'logout', 'session_114_1'),

(114, '2024-03-18', 'login', 'session_114_2'),
(114, '2024-03-18', 'search', 'session_114_2'),
(114, '2024-03-18', 'logout', 'session_114_2'),

(114, '2024-05-25', 'login', 'session_114_3'),
(114, '2024-05-25', 'add_to_cart', 'session_114_3'),
(114, '2024-05-25', 'logout', 'session_114_3'),

(114, '2024-05-29', 'login', 'session_114_4'),
(114, '2024-05-29', 'purchase', 'session_114_4'),
(114, '2024-05-29', 'logout', 'session_114_4'),

(114, '2024-06-14', 'login', 'session_114_5'),
(114, '2024-06-14', 'search', 'session_114_5'),
(114, '2024-06-14', 'logout', 'session_114_5'),

(115, '2024-01-21', 'login', 'session_115_1'),
(115, '2024-01-21', 'search', 'session_115_1'),
(115, '2024-01-21', 'logout', 'session_115_1'),

(115, '2024-03-02', 'login', 'session_115_2'),
(115, '2024-03-02', 'purchase', 'session_115_2'),
(115, '2024-03-02', 'logout', 'session_115_2'),

(115, '2024-04-25', 'login', 'session_115_3'),
(115, '2024-04-25', 'purchase', 'session_115_3'),
(115, '2024-04-25', 'logout', 'session_115_3'),

(115, '2024-05-24', 'login', 'session_115_4'),
(115, '2024-05-24', 'search', 'session_115_4'),
(115, '2024-05-24', 'logout', 'session_115_4'),

(115, '2024-05-29', 'login', 'session_115_5'),
(115, '2024-05-29', 'purchase', 'session_115_5'),
(115, '2024-05-29', 'logout', 'session_115_5'),

(115, '2024-06-20', 'login', 'session_115_6'),
(115, '2024-06-20', 'purchase', 'session_115_6'),
(115, '2024-06-20', 'logout', 'session_115_6'),

(116, '2024-01-24', 'login', 'session_116_1'),
(116, '2024-01-24', 'purchase', 'session_116_1'),
(116, '2024-01-24', 'logout', 'session_116_1'),

(116, '2024-02-17', 'login', 'session_116_2'),
(116, '2024-02-17', 'add_to_cart', 'session_116_2'),
(116, '2024-02-17', 'logout', 'session_116_2'),

(116, '2024-03-10', 'login', 'session_116_3'),
(116, '2024-03-10', 'search', 'session_116_3'),
(116, '2024-03-10', 'logout', 'session_116_3'),

(116, '2024-05-07', 'login', 'session_116_4'),
(116, '2024-05-07', 'search', 'session_116_4'),
(116, '2024-05-07', 'logout', 'session_116_4'),

(116, '2024-05-22', 'login', 'session_116_5'),
(116, '2024-05-22', 'add_to_cart', 'session_116_5'),
(116, '2024-05-22', 'logout', 'session_116_5'),

(116, '2024-06-02', 'login', 'session_116_6'),
(116, '2024-06-02', 'add_to_cart', 'session_116_6'),
(116, '2024-06-02', 'logout', 'session_116_6'),

(117, '2024-01-12', 'login', 'session_117_1'),
(117, '2024-01-12', 'search', 'session_117_1'),
(117, '2024-01-12', 'logout', 'session_117_1'),

(117, '2024-02-15', 'login', 'session_117_2'),
(117, '2024-02-15', 'add_to_cart', 'session_117_2'),
(117, '2024-02-15', 'logout', 'session_117_2'),

(117, '2024-03-17', 'login', 'session_117_3'),
(117, '2024-03-17', 'search', 'session_117_3'),
(117, '2024-03-17', 'logout', 'session_117_3'),

(117, '2024-04-17', 'login', 'session_117_4'),
(117, '2024-04-17', 'search', 'session_117_4'),
(117, '2024-04-17', 'logout', 'session_117_4'),

(117, '2024-06-07', 'login', 'session_117_5'),
(117, '2024-06-07', 'search', 'session_117_5'),
(117, '2024-06-07', 'logout', 'session_117_5'),

(117, '2024-06-19', 'login', 'session_117_6'),
(117, '2024-06-19', 'add_to_cart', 'session_117_6'),
(117, '2024-06-19', 'logout', 'session_117_6'),

(118, '2024-01-08', 'login', 'session_118_1'),
(118, '2024-01-08', 'search', 'session_118_1'),
(118, '2024-01-08', 'logout', 'session_118_1'),

(118, '2024-02-13', 'login', 'session_118_2'),
(118, '2024-02-13', 'search', 'session_118_2'),
(118, '2024-02-13', 'logout', 'session_118_2'),

(118, '2024-03-05', 'login', 'session_118_3'),
(118, '2024-03-05', 'search', 'session_118_3'),
(118, '2024-03-05', 'logout', 'session_118_3'),

(118, '2024-04-11', 'login', 'session_118_4'),
(118, '2024-04-11', 'purchase', 'session_118_4'),
(118, '2024-04-11', 'logout', 'session_118_4'),

(118, '2024-04-27', 'login', 'session_118_5'),
(118, '2024-04-27', 'add_to_cart', 'session_118_5'),
(118, '2024-04-27', 'logout', 'session_118_5'),

(118, '2024-05-21', 'login', 'session_118_6'),
(118, '2024-05-21', 'search', 'session_118_6'),
(118, '2024-05-21', 'logout', 'session_118_6'),

(118, '2024-06-25', 'login', 'session_118_7'),
(118, '2024-06-25', 'search', 'session_118_7'),
(118, '2024-06-25', 'logout', 'session_118_7'),

(119, '2024-02-15', 'login', 'session_119_1'),
(119, '2024-02-15', 'purchase', 'session_119_1'),
(119, '2024-02-15', 'logout', 'session_119_1'),

(119, '2024-03-30', 'login', 'session_119_2'),
(119, '2024-03-30', 'search', 'session_119_2'),
(119, '2024-03-30', 'logout', 'session_119_2'),

(119, '2024-05-06', 'login', 'session_119_3'),
(119, '2024-05-06', 'search', 'session_119_3'),
(119, '2024-05-06', 'logout', 'session_119_3'),

(119, '2024-05-26', 'login', 'session_119_4'),
(119, '2024-05-26', 'purchase', 'session_119_4'),
(119, '2024-05-26', 'logout', 'session_119_4'),

(120, '2024-01-11', 'login', 'session_120_1'),
(120, '2024-01-11', 'search', 'session_120_1'),
(120, '2024-01-11', 'logout', 'session_120_1'),

(120, '2024-01-15', 'login', 'session_120_2'),
(120, '2024-01-15', 'search', 'session_120_2'),
(120, '2024-01-15', 'logout', 'session_120_2'),

(120, '2024-06-10', 'login', 'session_120_3'),
(120, '2024-06-10', 'purchase', 'session_120_3'),
(120, '2024-06-10', 'logout', 'session_120_3');


-- -----------------------------------------------------------------------------
-- Create revenue_by_day — Rolling Metrics
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS revenue_by_day (
  revenue_date DATE PRIMARY KEY,
  revenue DECIMAL(10,2)
);

INSERT IGNORE INTO revenue_by_day (revenue_date, revenue) VALUES
('2024-01-09', 1338.47),
('2024-01-11', 293.28),
('2024-01-12', 182.25),
('2024-01-13', 58.69),
('2024-01-15', 594.25),
('2024-01-16', 517.45),
('2024-01-17', 362.31),
('2024-01-18', 24.32),
('2024-01-19', 302.91),
('2024-01-22', 262.82),
('2024-01-25', 446.98),
('2024-01-26', 0),
('2024-01-29', 283.96),
('2024-01-30', 482.20),
('2024-01-31', 220.76),
('2024-02-08', 99.24),
('2024-02-11', 230.50),
('2024-02-14', 501.82),
('2024-02-15', 292.75),
('2024-02-16', 894.23),
('2024-02-17', 647.96),
('2024-02-18', 392.70),
('2024-02-21', 411.22),
('2024-02-22', 325.76),
('2024-02-27', 136.82),
('2024-03-01', 1344.40),
('2024-03-02', 331.10),
('2024-03-03', 274.89),
('2024-03-04', 113.68),
('2024-03-05', 797.08),
('2024-03-06', 77.42),
('2024-03-07', 712.06),
('2024-03-08', 321.98),
('2024-03-09', 0),
('2024-03-10', 0),
('2024-03-11', 223.52),
('2024-03-12', 219.50),
('2024-03-13', 146.08),
('2024-03-14', 0),
('2024-03-15', 911.49),
('2024-03-16', 0),
('2024-03-17', 275.40),
('2024-03-18', 168.57),
('2024-03-19', 578.80),
('2024-03-21', 28.99),
('2024-03-25', 434.36),
('2024-03-26', 900.62),
('2024-03-27', 159.57),
('2024-03-30', 482.20),
('2024-03-31', 312.72),
('2024-04-03', 671.36),
('2024-04-04', 381.08),
('2024-04-09', 686.74),
('2024-04-10', 56.92),
('2024-04-11', 23.96),
('2024-04-12', 450.05),
('2024-04-14', 136.34),
('2024-04-18', 186.45),
('2024-04-19', 93.36),
('2024-04-24', 239.98),
('2024-04-26', 497.00),
('2024-04-28', 355.06),
('2024-05-02', 949.20),
('2024-05-07', 44.53),
('2024-05-08', 243.09),
('2024-05-10', 93.70),
('2024-05-22', 361.99),
('2024-05-23', 271.69),
('2024-05-24', 114.70),
('2024-05-25', 65.17),
('2024-05-26', 538.71),
('2024-05-27', 82.80),
('2024-05-30', 292.68),
('2024-06-03', 257.99),
('2024-06-06', 375.42),
('2024-06-07', 41.68),
('2024-06-08', 703.45),
('2024-06-09', 515.13),
('2024-06-11', 115.30),
('2024-06-13', 673.64),
('2024-06-15', 282.82),
('2024-06-16', 347.53),
('2024-06-20', 1823.75),
('2024-06-21', 68.34),
('2024-06-25', 74.51),
('2024-06-26', 142.52),
('2024-06-29', 57.14);



-- -----------------------------------------------------------------------------
-- Create product_events — Funnel Analysis
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS product_events (
  user_id INT,
  event_time DATETIME,
  product_id INT,
  event_type ENUM('view', 'add_to_cart', 'purchase')
);


INSERT IGNORE INTO product_events (user_id, event_time, product_id, event_type) VALUES
(108, '2024-03-14 09:15:00', 7, 'view'),
(115, '2024-03-07 10:10:45', 5, 'view'),
(104, '2024-06-12 15:12:30', 10, 'view'),
(116, '2024-03-10 12:20:00', 9, 'view'),
(113, '2024-03-01 14:05:15', 4, 'view'),
(116, '2024-06-02 08:30:00', 2, 'add_to_cart'),
(104, '2024-03-20 17:45:00', 6, 'purchase'),
(106, '2024-01-08 11:00:00', 2, 'view'),
(102, '2024-06-15 19:00:00', 6, 'view'),
(114, '2024-05-25 13:30:00', 6, 'add_to_cart'),

(113, '2024-05-09 09:30:00', 6, 'view'),
(104, '2024-03-12 10:45:00', 10, 'add_to_cart'),
(105, '2024-02-16 16:00:00', 5, 'view'),
(105, '2024-04-08 18:00:00', 9, 'view'),
(114, '2024-03-29 20:00:00', 4, 'purchase'),
(116, '2024-03-25 12:00:00', 7, 'view'),
(117, '2024-02-15 14:30:00', 7, 'add_to_cart'),
(104, '2024-05-01 08:00:00', 9, 'purchase'),
(106, '2024-04-02 21:00:00', 3, 'view'),
(120, '2024-03-14 22:00:00', 2, 'view'),

(118, '2024-02-13 10:00:00', 3, 'view'),
(106, '2024-06-19 14:30:00', 10, 'purchase'),
(102, '2024-06-28 13:00:00', 2, 'view'),
(102, '2024-05-13 09:45:00', 4, 'add_to_cart'),
(116, '2024-01-24 11:00:00', 5, 'purchase'),
(109, '2024-02-07 16:30:00', 6, 'view'),
(115, '2024-01-21 17:00:00', 2, 'view'),
(113, '2024-03-16 19:30:00', 2, 'add_to_cart'),
(115, '2024-05-29 20:00:00', 2, 'purchase'),
(114, '2024-03-18 10:00:00', 4, 'view'),

(103, '2024-04-13 08:00:00', 3, 'view'),
(101, '2024-03-26 14:00:00', 5, 'view'),
(107, '2024-04-09 13:00:00', 5, 'view'),
(115, '2024-03-02 12:30:00', 6, 'purchase'),
(117, '2024-02-20 16:00:00', 3, 'view'),
(113, '2024-06-08 18:30:00', 4, 'add_to_cart'),
(114, '2024-05-29 20:00:00', 4, 'purchase'),
(101, '2024-03-25 10:00:00', 2, 'view'),
(113, '2024-02-28 09:00:00', 6, 'view'),
(115, '2024-04-25 15:00:00', 1, 'purchase'),

(104, '2024-04-02 11:00:00', 9, 'view'),
(114, '2024-06-14 16:00:00', 10, 'view'),
(104, '2024-01-10 18:00:00', 2, 'add_to_cart'),
(101, '2024-02-26 08:30:00', 10, 'purchase'),
(104, '2024-03-25 14:00:00', 1, 'view'),
(101, '2024-06-24 09:30:00', 4, 'view'),
(103, '2024-04-03 12:00:00', 2, 'view'),
(118, '2024-05-21 19:00:00', 10, 'add_to_cart'),
(113, '2024-01-17 08:00:00', 2, 'view'),
(104, '2024-01-18 13:00:00', 6, 'purchase'),

(108, '2024-01-16 17:00:00', 7, 'view'),
(118, '2024-04-27 10:30:00', 6, 'add_to_cart'),
(116, '2024-02-21 15:00:00', 2, 'view'),
(117, '2024-03-17 16:30:00', 9, 'view'),
(118, '2024-04-11 18:00:00', 10, 'purchase'),
(119, '2024-03-30 20:00:00', 3, 'view'),
(118, '2024-01-08 14:00:00', 5, 'view'),
(120, '2024-06-10 12:30:00', 6, 'purchase'),
(108, '2024-06-05 13:00:00', 1, 'view'),
(117, '2024-06-19 09:00:00', 3, 'add_to_cart'),

(111, '2024-02-10 10:00:00', 2, 'view'),
(119, '2024-02-15 11:30:00', 9, 'purchase'),
(117, '2024-04-17 16:00:00', 8, 'view'),
(104, '2024-06-19 19:30:00', 9, 'add_to_cart'),
(110, '2024-01-30 18:00:00', 2, 'view'),
(109, '2024-01-14 20:00:00', 7, 'view'),
(119, '2024-05-06 14:00:00', 1, 'view'),
(107, '2024-06-19 16:00:00', 5, 'purchase'),
(104, '2024-04-25 12:30:00', 9, 'view'),
(109, '2024-03-14 10:00:00', 1, 'purchase'),

(101, '2024-01-08 09:00:00', 3, 'view'),
(103, '2024-03-11 11:00:00', 9, 'add_to_cart'),
(120, '2024-01-15 14:30:00', 10, 'view'),
(115, '2024-06-20 15:00:00', 6, 'purchase'),
(117, '2024-01-12 16:00:00', 4, 'view'),
(120, '2024-01-11 17:00:00', 10, 'view'),
(118, '2024-03-05 13:30:00', 8, 'view'),
(107, '2024-04-10 15:00:00', 10, 'add_to_cart'),
(104, '2024-03-06 09:00:00', 7, 'view'),
(101, '2024-01-28 08:00:00', 9, 'purchase'),

(106, '2024-02-29 10:00:00', 1, 'view'),
(111, '2024-04-23 11:00:00', 10, 'view'),
(102, '2024-04-18 12:00:00', 2, 'view'),
(115, '2024-05-24 13:00:00', 1, 'view'),
(102, '2024-03-24 14:00:00', 5, 'view'),
(103, '2024-03-03 15:00:00', 1, 'add_to_cart'),
(118, '2024-06-25 16:00:00', 4, 'view'),
(119, '2024-05-26 17:00:00', 4, 'purchase'),
(113, '2024-05-23 18:00:00', 9, 'view'),
(102, '2024-06-06 19:00:00', 6, 'purchase'),

(117, '2024-06-07 20:00:00', 1, 'view'),
(116, '2024-02-17 21:00:00', 10, 'add_to_cart'),
(114, '2024-03-14 22:00:00', 1, 'purchase'),
(104, '2024-04-29 23:00:00', 9, 'view'),
(109, '2024-03-04 09:00:00', 3, 'view'),
(111, '2024-06-08 10:00:00', 5, 'view'),
(113, '2024-02-14 11:00:00', 6, 'purchase'),
(116, '2024-05-22 12:00:00', 3, 'add_to_cart'),
(116, '2024-05-07 13:00:00', 4, 'view'),
(105, '2024-04-08 14:00:00', 6, 'view');


-- -----------------------------------------------------------------------------
-- Create traffic_sources — Conversion by Source
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS traffic_sources (
  user_id INT,
  source ENUM('organic', 'ads', 'email', 'referral', 'social'),
  signup_date DATE
);

INSERT IGNORE INTO traffic_sources (user_id, source, signup_date) VALUES
(1, 'organic', '2024-01-10'),
(2, 'ads', '2024-01-12'),
(3, 'email', '2024-01-15'),
(4, 'referral', '2024-01-20'),
(5, 'social', '2024-01-25'),
(6, 'organic', '2024-02-01'),
(7, 'ads', '2024-02-03'),
(8, 'email', '2024-02-07'),
(9, 'referral', '2024-02-10'),
(10, 'social', '2024-02-15'),
(11, 'organic', '2024-02-20'),
(12, 'ads', '2024-02-22'),
(13, 'email', '2024-02-25'),
(14, 'referral', '2024-03-01'),
(15, 'social', '2024-03-05'),
(16, 'organic', '2024-03-10'),
(17, 'ads', '2024-03-12'),
(18, 'email', '2024-03-15'),
(19, 'referral', '2024-03-18'),
(20, 'social', '2024-03-20'),
(21, 'organic', '2024-03-25'),
(22, 'ads', '2024-03-28'),
(23, 'email', '2024-04-01'),
(24, 'referral', '2024-04-05'),
(25, 'social', '2024-04-10'),
(26, 'organic', '2024-04-12'),
(27, 'ads', '2024-04-15'),
(28, 'email', '2024-04-18'),
(29, 'referral', '2024-04-20'),
(30, 'social', '2024-04-25'),
(31, 'organic', '2024-05-01'),
(32, 'ads', '2024-05-05'),
(33, 'email', '2024-05-08'),
(34, 'referral', '2024-05-10'),
(35, 'social', '2024-05-12'),
(36, 'organic', '2024-05-15'),
(37, 'ads', '2024-05-20'),
(38, 'email', '2024-05-22'),
(39, 'referral', '2024-05-25'),
(40, 'social', '2024-06-01'),
(41, 'organic', '2024-06-05'),
(42, 'ads', '2024-06-08'),
(43, 'email', '2024-06-10'),
(44, 'referral', '2024-06-12'),
(45, 'social', '2024-06-15'),
(46, 'organic', '2024-06-18'),
(47, 'ads', '2024-06-20'),
(48, 'email', '2024-06-22'),
(49, 'referral', '2024-06-25'),
(50, 'social', '2024-06-28');


CREATE TABLE IF NOT EXISTS conversions (
  user_id INT,
  conversion_date DATE,
  conversion_type ENUM('signup', 'purchase')
);


INSERT IGNORE INTO conversions (user_id, conversion_date, conversion_type) VALUES
(1, '2024-01-10', 'signup'),
(2, '2024-01-12', 'signup'),
(3, '2024-01-15', 'signup'),
(4, '2024-01-20', 'signup'),
(5, '2024-01-25', 'signup'),
(6, '2024-02-01', 'signup'),
(7, '2024-02-03', 'signup'),
(8, '2024-02-07', 'signup'),
(9, '2024-02-10', 'signup'),
(10, '2024-02-15', 'signup'),
(11, '2024-02-20', 'signup'),
(12, '2024-02-22', 'signup'),
(13, '2024-02-25', 'signup'),
(14, '2024-03-01', 'signup'),
(15, '2024-03-05', 'signup'),
(16, '2024-03-10', 'signup'),
(17, '2024-03-12', 'signup'),
(18, '2024-03-15', 'signup'),
(19, '2024-03-18', 'signup'),
(20, '2024-03-20', 'signup'),
(21, '2024-03-25', 'signup'),
(22, '2024-03-28', 'signup'),
(23, '2024-04-01', 'signup'),
(24, '2024-04-05', 'signup'),
(25, '2024-04-10', 'signup'),
(26, '2024-04-12', 'signup'),
(27, '2024-04-15', 'signup'),
(28, '2024-04-18', 'signup'),
(29, '2024-04-20', 'signup'),
(30, '2024-04-25', 'signup'),
(31, '2024-05-01', 'signup'),
(32, '2024-05-05', 'signup'),
(33, '2024-05-08', 'signup'),
(34, '2024-05-10', 'signup'),
(35, '2024-05-12', 'signup'),
(36, '2024-05-15', 'signup'),
(37, '2024-05-20', 'signup'),
(38, '2024-05-22', 'signup'),
(39, '2024-05-25', 'signup'),
(40, '2024-06-01', 'signup'),
(41, '2024-06-05', 'signup'),
(42, '2024-06-08', 'signup'),
(43, '2024-06-10', 'signup'),
(44, '2024-06-12', 'signup'),
(45, '2024-06-15', 'signup'),
(46, '2024-06-18', 'signup'),
(47, '2024-06-20', 'signup'),
(48, '2024-06-22', 'signup'),
(49, '2024-06-25', 'signup'),
(50, '2024-06-28', 'signup'),
-- Some purchase conversions after signup
(1, '2024-01-20', 'purchase'),
(3, '2024-01-25', 'purchase'),
(5, '2024-02-01', 'purchase'),
(7, '2024-02-15', 'purchase'),
(10, '2024-02-28', 'purchase'),
(15, '2024-03-20', 'purchase'),
(18, '2024-03-30', 'purchase'),
(22, '2024-04-10', 'purchase'),
(25, '2024-04-20', 'purchase'),
(28, '2024-04-28', 'purchase'),
(30, '2024-05-05', 'purchase'),
(33, '2024-05-15', 'purchase'),
(37, '2024-05-30', 'purchase'),
(40, '2024-06-10', 'purchase'),
(45, '2024-06-25', 'purchase'),
(50, '2024-07-05', 'purchase');


-- -----------------------------------------------------------------------------
-- Create performance_logs — SLA / Outlier Detection
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS performance_logs (
  service_name VARCHAR(50),
  log_time DATETIME,
  response_time_ms INT,
  status_code INT
);

INSERT IGNORE INTO performance_logs (service_name, log_time, response_time_ms, status_code) VALUES
('auth_service', '2024-06-01 08:15:23', 150, 200),
('auth_service', '2024-06-01 08:17:45', 135, 200),
('auth_service', '2024-06-01 09:05:12', 950, 500),
('auth_service', '2024-06-01 09:30:01', 180, 200),
('auth_service', '2024-06-01 10:45:23', 220, 200),

('payment_service', '2024-06-01 08:16:44', 340, 200),
('payment_service', '2024-06-01 08:18:59', 1200, 503),
('payment_service', '2024-06-01 09:20:33', 870, 200),
('payment_service', '2024-06-01 10:10:12', 400, 429),
('payment_service', '2024-06-01 11:00:01', 200, 200),

('product_service', '2024-06-01 08:25:10', 80, 200),
('product_service', '2024-06-01 08:50:17', 60, 200),
('product_service', '2024-06-01 09:05:45', 70, 404),
('product_service', '2024-06-01 09:15:22', 150, 200),
('product_service', '2024-06-01 09:55:30', 2000, 500),

('user_service', '2024-06-01 08:10:05', 90, 200),
('user_service', '2024-06-01 08:30:45', 110, 200),
('user_service', '2024-06-01 09:00:15', 95, 200),
('user_service', '2024-06-01 09:45:50', 150, 429),
('user_service', '2024-06-01 10:25:05', 180, 200),

('auth_service', '2024-06-02 08:12:33', 130, 200),
('payment_service', '2024-06-02 08:14:12', 600, 200),
('product_service', '2024-06-02 08:40:23', 50, 200),
('user_service', '2024-06-02 09:10:59', 120, 200),

('auth_service', '2024-06-02 10:05:11', 175, 200),
('payment_service', '2024-06-02 10:15:24', 1020, 500),
('product_service', '2024-06-02 10:30:35', 95, 200),
('user_service', '2024-06-02 11:00:05', 200, 200);



-- -----------------------------------------------------------------------------
-- Create customers — Segmentation
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS customers (
  customer_id INT PRIMARY KEY,
  signup_date DATE,
  country VARCHAR(50),
  total_spend DECIMAL(10,2),
  num_orders INT,
  last_order_date DATE
);

INSERT IGNORE INTO customers (customer_id, signup_date, country, total_spend, num_orders, last_order_date) VALUES
(101, '2023-11-05', 'United States', 1050.75, 12, '2024-06-25'),
(102, '2023-12-12', 'Canada', 785.40, 8, '2024-06-29'),
(103, '2024-01-18', 'United Kingdom', 540.60, 5, '2024-06-15'),
(104, '2023-10-20', 'Germany', 2300.90, 25, '2024-06-20'),
(105, '2024-02-05', 'France', 420.00, 4, '2024-04-09'),
(106, '2024-03-15', 'Australia', 1325.25, 15, '2024-06-07'),
(107, '2024-01-28', 'Brazil', 655.80, 7, '2024-04-11'),
(108, '2023-12-22', 'India', 980.40, 10, '2024-06-06'),
(109, '2024-03-05', 'Japan', 310.00, 3, '2024-03-15'),
(110, '2024-01-11', 'South Africa', 580.50, 6, '2024-01-31'),
(111, '2023-11-30', 'Mexico', 1250.10, 14, '2024-06-09'),
(112, '2024-02-20', 'Netherlands', 770.90, 8, '2024-04-30'),
(113, '2024-01-09', 'Italy', 900.25, 9, '2024-06-24'),
(114, '2023-10-05', 'Spain', 1650.00, 18, '2024-06-15'),
(115, '2024-04-01', 'Sweden', 450.30, 5, '2024-05-30'),
(116, '2023-11-15', 'Switzerland', 2000.85, 23, '2024-06-26'),
(117, '2024-02-28', 'Norway', 300.00, 3, '2024-06-20'),
(118, '2023-12-05', 'Denmark', 1550.40, 17, '2024-06-28'),
(119, '2024-03-19', 'New Zealand', 410.75, 4, '2024-05-27'),
(120, '2024-01-25', 'Ireland', 640.00, 6, '2024-06-25');



-- -----------------------------------------------------------------------------
-- Create sales_history — Forecasting
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sales_history (
  sale_date DATE,
  product_id INT,
  units_sold INT,
  revenue DECIMAL(10,2)
);

INSERT IGNORE INTO sales_history (sale_date, product_id, units_sold, revenue) VALUES
('2024-06-01', 1, 15, 1517.10),
('2024-06-01', 2, 10, 1345.80),
('2024-06-01', 3, 8, 1027.20),
('2024-06-01', 4, 12, 1986.00),
('2024-06-01', 5, 6, 1079.94),
('2024-06-02', 1, 18, 1818.28),
('2024-06-02', 2, 9, 1211.61),
('2024-06-02', 3, 7, 897.50),
('2024-06-02', 4, 10, 1655.00),
('2024-06-02', 5, 5, 899.95),
('2024-06-03', 1, 20, 2024.00),
('2024-06-03', 2, 14, 1871.60),
('2024-06-03', 3, 12, 1530.00),
('2024-06-03', 4, 11, 1812.50),
('2024-06-03', 5, 8, 1439.92),
('2024-06-04', 1, 22, 2223.20),
('2024-06-04', 2, 13, 1737.40),
('2024-06-04', 3, 9, 1147.50),
('2024-06-04', 4, 15, 2475.00),
('2024-06-04', 5, 10, 1799.90),
('2024-06-05', 1, 17, 1717.30),
('2024-06-05', 2, 12, 1608.00),
('2024-06-05', 3, 11, 1402.50),
('2024-06-05', 4, 13, 2147.50),
('2024-06-05', 5, 7, 1259.93);
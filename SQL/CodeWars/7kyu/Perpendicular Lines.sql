-- 7kyu Fundamentals, Geometry, Algorithms

-- You are given an input (n) which represents the amount of lines you are given, your job is to figure out what is the maximum amount of perpendicular lines you can make using these lines.

-- Note: A perpendicular line is one that forms a 90 degree angle

-- n will always be greater than or equal to 0

--# write your SQL statement here: 
-- you are given a table 'perpendicular' with column 'n'
-- return a table with this column and your result in a column named 'res'.

-- 3 2
-- 4 4 
-- 5 6
-- 6 9 n^2/4 = perpendicular lines

SELECT n, CAST(FLOOR(n * n / 4) AS INTEGER) AS res 
FROM perpendicular

-- Alternative answer that alters table

ALTER TABLE perpendicular
DROP COLUMN IF EXISTS res;

ALTER TABLE perpendicular
ADD COLUMN res INT;

UPDATE perpendicular
SET res = FLOOR(n^2 / 4);

SELECT n, res FROM perpendicular;

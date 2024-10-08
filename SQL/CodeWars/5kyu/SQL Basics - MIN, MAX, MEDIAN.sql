-- 6kyu SQL Basics - MIN, MAX and MEDIAN

-- For this challenge you need to create a simple SELECT statement. Your task is to calculate the MIN, MEDIAN and MAX scores of the students from the results table.

-- Resultant table:
-- min
-- median
-- max

SELECT 
  MIN(score), 
  percentile_cont(0.5) WITHIN GROUP (ORDER BY score) AS median, 
  MAX(score) 
FROM result

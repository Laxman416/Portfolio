# SQL and Git Fundamentals <!-- omit in toc -->

**Course on Datacamp:**<br> 
`'Introduction to SQL'`<br> 
`'Intermediate SQL'`<br> 
`'Joining Data in SQL'`<br> 
`'Introduction to Git'`<br> 

- [Introduction to SQL](#introduction-to-sql)
  - [Relational Databases](#relational-databases)
  - [Querying](#querying)
- [Intermediate SQL](#intermediate-sql)
  - [Selecting Data](#selecting-data)
  - [Filtering Records](#filtering-records)
  - [Aggregate Functions](#aggregate-functions)
  - [Sorting and Grouping](#sorting-and-grouping)
- [Joining Data in SQL](#joining-data-in-sql)
  - [Introduction Inner Joins](#introduction-inner-joins)
  - [Outer Joins, Cross Joins, and Self Joins](#outer-joins-cross-joins-and-self-joins)
  - [Set Theory for SQL Joins](#set-theory-for-sql-joins)
  - [Subqueries](#subqueries)
- [Introduction to Git](#introduction-to-git)
  - [Introduction](#introduction)
  - [Making changes](#making-changes)
  - [Git workflows](#git-workflows)
  - [Collaborating](#collaborating)


# Introduction to SQL

## Relational Databases

**In this chapter:**
- Databases
- Tables
- Data

Database:
- Multiple tables
- Relational databases: defines relationships between tables
- More storage and secure
- Querying from multiple people
- Best large databases

Tables:
- records (row) and columns (fields)
- lower case table names
- field name singular and lowercase
- unique identifier: id
  
Data:
- each field has same data type
- VARCHAR(255)
- INT
- NUMERIC (float)

Schemas:
- Blueprint shows relationships between databases and data type of each column

Database saved in server.

## Querying

**In this chapter:**
- Queries
- SQL flavours

Keywords:
- `SELECT`
- `FROM`
- `AS`
- `SELECT DISTINCT`
- `CREATE VIEW`

```SQL
CREATE VIEW virtual_table AS
SELECT DISTINCT column1 AS c1 ,column2
FROM table_name
```

View: virtual table as a result of SQL statement

**SQL flavors**

- PostgreSQL:  `LIMIT`
- SQL Server: `SELECT TOP(2)`

# Intermediate SQL

## Selecting Data

**In this chapter:**
- Querying Database
- Query execution
- SQL style

Keywords:
- `COUNT`

```SQL
SELECT COUNT(DISTINCT column1), COUNT(*) AS total_records
FROM table_name
```

**Execution:**
```SQL
SELECT name -- Third
FROM people -- First executed
WHERE condition -- Second
LIMIT 10; -- Final
```

**SQL Formatting**

- Capital Keywords and newlines
- could indent each field
- 
## Filtering Records

**In this chapter:**
- Filtering numbers
- Multiple criteria
- Filtering text
- NULL values

**Filter**

Keywords:
- `WHERE`
- `<>` not equal to
  
```SQL
SELECT column
FROM table_name
WHERE column condition;
```
**Multiple Criteria**

Keywords:
- `OR`
- `AND`
- `BETWEEN 1 AND 5`

**Filtering Text**

Keywords:
- `LIKE`
  - `%`: match zero, one, or many characters
  - `_`: match single character
- `NOT LIKE`
- `IN`  

```SQL
SELECT column1
FROM table_name
WHERE column1 LIKE 'Ade%';
```

```SQL
SELECT column1
FROM table_name
WHERE column1 IN (1920,1930,1940)
```

**NULL values**

- COUNT only includes non-missing values
- `IS NULL`
- `IS NOT NULL`

```sql
SELECT COUNT(*)
FROM table_name
WHERE column1 IS NULL
```

## Aggregate Functions

**In this chapter:**
- Summarising Data
- Summarising Subsets
- Aliasing and Arithmetic

**Summarising Data**

Key Words Fn:
- `AVG()`
- `SUM()`
- `MAX()`
- `MIN()`
- `COUNT()`
- `ROUND(, 2)`
- `ROUND(, -5)` --> 100,000 ROUNDING
**Summarising Subsets**

```SQL
SELECT AVG(column1)
FROM table_name
WHERE column2 >= 2020
```

**Aliasing and Arithmetic**

Key Words:
- `+`, `-`, `/`, `*`

Cant use Alias in `WHERE` due to order of execution

## Sorting and Grouping

**In this chapter:**
- Sorting
- Grouping
- Filtering grouped data

**Sorting**

Key Words:
- `ORDER BY ... ASC/DESC` 

```sql
SELECT
FROM
ORDER BY column1 DESC, column2 DESC
```
**Grouping**

Key Words:
- `GROUP BY ...`

```SQL
SELECT
FROM
GROUP BY column1, column2
ORDER BY
LIMIT
```
**Filtering**

Key Words:
`HAVING`

```SQL
SELECT
FROM
WHERE
GROUP BY column1, column2
HAVING condition
ORDER BY
LIMIT
```

# Joining Data in SQL

## Introduction Inner Joins

**In this chapter:**
- INNER JOIN
- Defining relationships
- Multiple joins

**INNER JOIN**
- key field unique identifier for each record
- `INNER JOIN` looks for records in both tables that match

```SQL
SELECT t1.column, t2.column
FROM table1 AS t1
INNER JOIN table2 AS t2
ON t1.id = t2.id;

INNER JOIN tabl2
USING(identical_column_name)
```

**Defining Relationships**

Three types of relationships in databases:
 - **One to One:** Each record(row) in one table associated with another record in another table.
 - **One to Many:** each record in one table associated with multiple records in another table/
 - **Many to Many:** records in both tables can be associated with multiple records in the other table.
  
**Multiple Joins**

```SQL
SELECT *
FROM left_table
INNER JOIN right_table
USING(id)
INNER JOIN another_table
USING(id)
```

Joining on multiple keys:
```SQL
SELECT *
FROM left_table
INNER JOIN right_table
ON left_table.id = right_table.id
INNER JOIN another_table
ON left_table.date = right_table.date
```

## Outer Joins, Cross Joins, and Self Joins

**In this chapter:**
- LEFT and RIGHT JOIN
- FULL JOIN
- CROSS JOIN
- SELF JOIN

**LEFT JOIN and RIGHT JOIN**

LEFT JOIN:  All records on left table + matching tables on right
RIGHT JOIN: All records on right table + matching tables on left

```SQL
SELECT
FROM
LEFT JOIN table_name2
USING(column);
```

**FULL JOIN**

FULL JOIN: LEFT and RIGHT join together

```SQL
SELECT
FROM
FULL JOIN table_name2
USING(id);
```

**Self join**

Comparing parts of same table

```
SELECT p1.country AS country1,
       p2.country AS country2,
       p1.continent
From p1
INNER JOIN p2
ON p1.continent = p2.continent
    AND p1.country <> p2.country;
```

## Set Theory for SQL Joins

**In this chapter:**
- Set theory for SQL Joins
- INTERSECT
- EXCEPT

**Set Theory**

3 Set Opertations:
- `UNION`: two tables and returns all record from tables not including duplicates
- `UNION ALL`: includes the duplicates as well
- `INTERSECT`:
- `EXCEPT`

![SET Operations](../Images/Joining%20Data%20in%20SQL/Set%20Operatoins.png)


```SQL
SELECT *
FROM table_1
UNION
SELECT *
FROM table_2
```

**INTERSECT**

- needs same number of columns
- returns records that have all fields that match

**EXCEPT**

- returns only records in left, returns none in both

## Subqueries

**In this chapter:**
- Subquerying with semi joins and anti joins
- Subquery inside WHERE and SELECT
- Subquery inside FROM

**SEMI JOINS and ANTI JOINS**
 
SEMI JOIN: choose records in first table where col1 in right col2
ANTI JOIN: returns records in first table where values are not in second column

SEMI JOIN:
```SQL
SELECT field
FROM t1
WHERE field IN
    (SELECT field
     FROM t2
     WHERE condition);
```

ANTI JOIN:
```SQL
SELECT field
FROM t1
WHERE field NOT IN
    (SELECT field
     FROM t2
     WHERE condition);
```

**Subquery with WHERE AND SELECT**

`WHERE field1 IN (SELECT field2)`
- field1 and field2 have to be same datatype

Subquery instead of JOIN
```SQL
SELECT DISTINCT continent,
    (SELECT COUNT(*)
     FROM t2
     WHERE t1.continent = t2.continent) AS field_name
FROM t1
```

**Subquery with FROM**

```SQL
-- Query to return continents with monarchs and the year of most recent country independence gained
SELECT DISTINCT continents, sub.most_recent
FROM t1,
    (SELECT
        continent,
        MAX(indep_year) AS most_recent
     FROM t2
     GROUP BY continent) AS sub
WHERE t1.id = sub.id
```

# Introduction to Git

## Introduction

**In this chapter:**
- Versions
- Saving Files
- Comparing Files

**Versions**

- Version control used to track and allow simultaneous development

Shell commands:
- `pwd`
- `ls`
- `cd`
- `nano` save with Ctrl + O and exit with Ctrl + X
- `echo` crete files

**Saving files**

Staging: `git add` `git add .`
Commiting `git commit -m ''`

![Staging Area Diagram](../Images/Git/Git%20Workflow.png)

**Comparing files**

`git diff file_name ` compare files not in staging area with comitted
`git diff -r HEAD file_name` compare files in staging area with last commit

## Making changes

**In this chapter:**
- Storing data with GIT
- Viewing changes
- Undoing changes before committing
- Restoring and reverting

Commit Structure:
- Commit: metadata
- Tree: tracks names and locations in repo 
- Blob: for each file: Binary large object
  - Contain compressed screenshot of file's content

`Git log` shows hash
`git show hash#` 

**Viewing changes**

`git diff -r HEAD~1` second most recent commit
`git show HEAD~1`
`git diff HEAD~1 HEAD~2`

`git annotate file`</br>
- Hash, Author, Time, Line#, Line Content

**Undoing changes**

Unstage file:
- `git reset HEAD summary_statistics.csv`
- `git reset HEAD`

Undo changes to an unstaged file:
`git checkout -- file_name`
`git checkout .`

**Restoring and reverting**

Use commit history using git log

`git log -3`
`git log --since = 'Month Day Year' --until = 'Month Day Year'`

Revert:
- `git checkout HEAD~1/HASH# filename`
- `git checkout HEAD~1/HASH#`

Clean:
- `git clean - n` List untracked files
- `git clean -f` Untracked files cleaned

## Git workflows

**In this chapter:**
- Configuring Git
- Branches
- Switching/Merging
- Handling conflict

**Configuring Git**

`git config --list`
`git config --global alias.st status`

3 levels:
- `local`: 1 project
- `global`: all projects
- `system`: every user on pc

**Branches:**

![Branches](../Images/Git/branches.png)
Branch for specific task

- `git branch`
- `git checkout -b branch_name` Create branches
- `git diff main branch_name`

Switching:
- `git checkout branch_name`

Merge:
- `git merge source destination`

Handling conflict:
- File in different branches contain different contain that cant be merged


![Conflicts](../Images/Git/Conflicts%20merge.png)

- <<<<<< HEAD -> implies in current branch
- line with equal signs centre of conflict
- after means different content on same line
- >>>>> branch -> other branch

Prevent:
- each branch for specific task

## Collaborating

**In this chapter:**
- Creating repos
- Working with remotes
- Gathering from remote
- Pushing to remote

**Creating**

- `git init directory` Create news git folder
- `git init` Existing project:

Avoid Nested repositories

**Remote Repo**

- stored on cloud
- `git clone path_to_project repo_name`
- `git clone URL`

**Gathering Remote**

- `git fetch origin main` name of remote to local branch
- `git fetch origin branch_name`

- `git merge origin main`

- `git pull origin main`: fetch and merge. pull origin to local main branch

Can't pull with unsaved local changes

**Pushing**
`git push origin main`

if errors: pull with message on txt file</br>
then push again
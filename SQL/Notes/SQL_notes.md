# Introduction to SQL <!-- omit in toc -->

SQL is used for managing and manipulating databases. Allows to store, retrieve, update and delete data.
MySQL Workbench is an app for SQL.

- [Database Basics](#database-basics)
- [SQL Keywords](#sql-keywords)
- [SQL Syntax](#sql-syntax)
- [Creating Database/tables](#creating-databasetables)
- [Sorting Data](#sorting-data)
- [Foreign Key / Table Relationships](#foreign-key--table-relationships)
- [Modifying Database](#modifying-database)
  - [Inserting Data](#inserting-data)
  - [Updating Data](#updating-data)
  - [Deleting Data](#deleting-data)
- [Data Constraints](#data-constraints)
- [Querying Data](#querying-data)
- [Joins and Relationships](#joins-and-relationships)
  - [JOIN Statement](#join-statement)

## Database Basics
- Rows known as records
- Columns known as fields
- Columns define type of data stored
- Databases can have tables connected with each other - Relational Database.

## SQL Keywords

- **SELECT:** Retrieves data from table/tables
- **INSERT:** Adds new data
- **UPDATE:** Modifies data
- **DELETE:** Removes data
- **FROM:** Specifies table
- **WHERE:** Conditional statement
- **CREATE:** `CREATE DATABASE name_of_database` create table or database
- **ORDER BY:** Sorting
- **DROP:** Delete database
- **USE:** `USE database_name` specifies which database to use
- **INSERT INTO:** Inserting data into database. Specifies table and columns.
- **VALUES:** The values you want to insert.
- **UPDATE:** Specifies table you want to update.
- **SET:**  Specifies the values you want to update in the table.

## SQL Syntax
```sql
SELECT column1, column2, row1, ...  
FROM table_name 
WHERE condition;
```
- Need Semicolon at end

## Creating Database/tables
```sql
CREATE DATABASE database_name
USE database_name
Create TABLE test (
id INT NOT NULL AUTO_INCREMENT,
column1 INT NOT NULL,
PRIMARY KEY(id)
);

ALTER TABLE test
ADD column2 VARCHAR(255);
```
- **ALTER:** keyword to add columns/ rows
- **VARCHAR(255):** string 
- **NOT NULL:** requires it to be not empty
- **AUTO_INCREMENT:** used for primary key 
- **PRIMARY KEY:** used to define primary key
## Sorting Data

```sql
SELECT column1, column2, row1, ...  
FROM table_name 
WHERE condition
ORDER BY column1 ASC, column2 DESC;
```
- **ASC:** Ascending
- **DESC:** Descending

## Foreign Key / Table Relationships
Example: bands and albums table. Albums could have a column for band_id which is `FOREIGN KEY` that references the primary id of the band table.

```sql
CREATE TABLE ablums (
    id INT NOT NULL AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    band_id INT NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (band_id) REFERENCES band(id)
);
```
 - `FOREIGN KEY (column_name) REFERENCES table_name(column_name)`


## Modifying Database

 Modify the data stored in database. Section covers inserting, updating and deleting data.

### Inserting Data
Inserts a new row into `table_name`

```sql
INSERT INTO table_name (column1, column2, ...)
VALUES (value1, value2), ...;
```

### Updating Data
Updates the values of columns

```sql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
WHERE condition;
```
### Deleting Data

```sql
DELETE FROM table_name
WHERE condition;
```

## Data Constraints

Constraints
- **PRIMARY KEY:** Used to identify each row.
- **FOREIGN KEY:** Establishes a relationship between two tables.
- **Unique Constraint:** Ensures the uniqueness of values in column/s
- **Check Constraint:** Defines a condition that must be true for row to be valid.

## Querying Data

- **Select All:** `SELECT * FROM table_name`

- **LIMIT Select:** `SELECT * FROM table_name LIMIT 2`
- **Select from specific column** `SELECT column_name FROM table_name` 
- **As** `SELECT id AS 'ID', name AS 'Band Name'`

- **ORDER BY:** `SELECT * FROM table_name ORDER BY name`

- **DISTINCT:** `SELECT DISTINCT column_name FROM albums` - Only shows distinct values

- **LIKE String Filter** `WHERE name LIKE '%er%'` - % represents any amount of characters.

- **OR:** `WHERE column1 = 1 OR column2 = 2;`
  
- **AND:** `WHERE year = 10 AND band_id = 1;`

- **BETWEEN:** `WHERE year BETWEEN 10 AND 20;`

## Joins and Relationships
 Three types of relationships in databases:
 - **One to One:** Each record(row) in one table associated with another record in another table.
 - **One to Many:** each record in one table associated with multiple records in another table/
 - **Many to Many:** records in both tables can be associated with multiple records in the other table.

### JOIN Statement
3 types of JOIN statements: `INNER JOIN, LEFT JOIN, RIGHT JOIN AND FULL JOIN`

- **INNER JOIN**
Retrieves rows that have matching values in both tables being joined.

- **JOIN Statements**







---

**Author:** Laxman Seelan

**Credits:** s-shemmee(github) and Web Dev Simplified (Youtube)
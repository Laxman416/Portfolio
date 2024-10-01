# Introduction to SQL

SQL is used for managing and manipulating databases. Allows to store, retrieve, update and delete data.
MySQL Workbench is an app for SQL.

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
- **CREATE:** `CREATE DATABASE name_of_database`
- **ORDER BY:** Sorting
## SQL Syntax
```sql
SELECT column1, column2, row1, ...  
FROM table_name 
WHERE condition;
```
- Need Semicolon at end

### Sorting Data

```sql
SELECT column1, column2, row1, ...  
FROM table_name 
WHERE condition
ORDER BY column1 ASC, column2 DESC;
```
- **ASC:** Ascending
- **DESC:** Descending

---

**Author:** Laxman Seelan
**Credits:** s-shemmee(github) and Web Dev Simplified (Youtube)
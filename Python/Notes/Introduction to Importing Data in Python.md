# Introduction to Importing Data in Python
- [Introduction to Importing Data in Python](#introduction-to-importing-data-in-python)
  - [Introduction and flat files](#introduction-and-flat-files)
    - [Flat Files (csv/txt)](#flat-files-csvtxt)
  - [Importing data from other file type](#importing-data-from-other-file-type)
  - [Working with Relational databases in Python](#working-with-relational-databases-in-python)
    - [Relational Databases](#relational-databases)
    - [Querying Relational Databases with pandas](#querying-relational-databases-with-pandas)
    - [Advanced Querying:](#advanced-querying)

Course from DataCamp

## Introduction and flat files

import data from txt and csv
files from other software
relational databases

plain texts or table data

**.txt:**
```
with open('', mode = 'r/w') as filename:
    print(file.read())
    print(file.readline())

```

### Flat Files (csv/txt)
text files containing records -> table data
Record: row of fields
Columns: feature

can have header
Delimiter

store in NumPy or Pandas

**Importing using NumPy:**

**Parameter:**
- `skiprows = int`
- `delimiter = ''`
- `dtype = str/int`

```python

data = np.loadtxt(filename, delimiter=',', skiprows)

```


**Importing using Pandas:**

```python

data = np.read_csv(filename, skiprows)

```

## Importing data from other file type

**Importing using Pickled files:**
```python
with open('', 'rb') as filename: #read only binary
    data = pickle.load(filename)

```

**Importing Excel files:**

**Parameter/Fn:**
- `data.sheet_names`
- `data.parse(sheet_name/index)`
- parameter: `name`
  
```python

data = pd.ExcelFile(filename)

```

**Importing SAS files:**
Statistics

**Parameter/Fn:**
- data.sheet_names
- data.parse(sheet_name/index)

```python
from sas7bdat import SAS7BDAT

with SAS7BDAT('') as file:
    df_sas = file.to_data_frame()
```

**Importing Stata files:**
Statistics

```python

data = pd.read_stata('.dta')
```

**Importing HDF5 files:**
Large number of data

```python
import h5py

data = h5py.File('.mat'. 'r')

for key in data.keys():
    print(key)
```
meta
quality
strain

pass keys to numpy array to read

`np.array(data['meta']['Description'])`

**Importing MatLab files:**
Matrix capabilitys

```python
import scipy.io

data = scipy.io.loadmat('.mat')
```

## Working with Relational databases in Python

### Relational Databases
- table has foreign key relating to different table
- needs Primary Key
- SQL

**Connecting to Database**

```python
from sqlachemy import create_engine
engine = create_engine('sqlite:///.sqlite')

table_names = engine.table_names()

```

**Querying Data:**

```python
from sqlalchemy import create_engine
engine = create_engine('sqlite:///.sqlite')

with engine.connect() as con:
    rs = con.execute("SELECT * FROM Orders")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

```

### Querying Relational Databases with pandas

```python
from sqlachemy import create_engine
engine = create_engine('sqlite:///.sqlite')

df = pd.read_sql_query("SELECT * FROM Orders", engine)

```

### Advanced Querying:
 `INNER JOIN Customers on column1 = column2, engine)`


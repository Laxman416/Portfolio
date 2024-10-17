# Introduction to Importing Data in Python

**Course on Datacamp:**<br> 
`'Introduction to Importing Data in Python'`<br> 
`'Cleaning Data in Python'`<br>

- [Introduction to Importing Data in Python](#introduction-to-importing-data-in-python)
  - [Introduction and flat files](#introduction-and-flat-files)
    - [Flat Files (csv/txt)](#flat-files-csvtxt)
  - [Importing data from other file type](#importing-data-from-other-file-type)
  - [Working with Relational databases in Python](#working-with-relational-databases-in-python)
    - [Relational Databases](#relational-databases)
    - [Querying Relational Databases with pandas](#querying-relational-databases-with-pandas)
    - [Advanced Querying:](#advanced-querying)
- [Cleaning Data](#cleaning-data)
  - [Common Data Problems](#common-data-problems)
    - [Data Type Constraints](#data-type-constraints)
    - [Data Range Constraints](#data-range-constraints)
    - [Unique Constraints](#unique-constraints)
  - [Text and Categorical Problems](#text-and-categorical-problems)
    - [Membership Constraints](#membership-constraints)
    - [Categorical Variables](#categorical-variables)
    - [Cleaning Text Data](#cleaning-text-data)
  - [Advanced Data Problems](#advanced-data-problems)
    - [Uniformity](#uniformity)
    - [Cross Field Validation](#cross-field-validation)
    - [Completeness](#completeness)
  - [Record Linkage](#record-linkage)
    - [Comparing Strings](#comparing-strings)
    - [Generating Pairs](#generating-pairs)
    - [Linking df](#linking-df)

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


df = pd.read_picke('') # pandas pickle
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

# Cleaning Data


- [Introduction to Importing Data in Python](#introduction-to-importing-data-in-python)
  - [Introduction and flat files](#introduction-and-flat-files)
    - [Flat Files (csv/txt)](#flat-files-csvtxt)
  - [Importing data from other file type](#importing-data-from-other-file-type)
  - [Working with Relational databases in Python](#working-with-relational-databases-in-python)
    - [Relational Databases](#relational-databases)
    - [Querying Relational Databases with pandas](#querying-relational-databases-with-pandas)
    - [Advanced Querying:](#advanced-querying)
- [Cleaning Data](#cleaning-data)
  - [Common Data Problems](#common-data-problems)
    - [Data Type Constraints](#data-type-constraints)
    - [Data Range Constraints](#data-range-constraints)
    - [Unique Constraints](#unique-constraints)
  - [Text and Categorical Problems](#text-and-categorical-problems)
    - [Membership Constraints](#membership-constraints)
    - [Categorical Variables](#categorical-variables)
    - [Cleaning Text Data](#cleaning-text-data)
  - [Advanced Data Problems](#advanced-data-problems)
    - [Uniformity](#uniformity)
    - [Cross Field Validation](#cross-field-validation)
    - [Completeness](#completeness)
  - [Record Linkage](#record-linkage)
    - [Comparing Strings](#comparing-strings)
    - [Generating Pairs](#generating-pairs)
    - [Linking df](#linking-df)

## Common Data Problems

- Access Data - Explore - Extract - Report Insights
 
### Data Type Constraints
string to int:

`df['].str.strip()`

check if int:
`assert df[''].dtype == 'int'`

### Data Range Constraints

Dropping data, only when small
Setting custom min/max
Treat as missing
Setting custom values

```python

df = df[df[''] > 5] # Drop using filtering
df.drop(df[df[''] > 5].index, inplace = True) # Drop using .drop

assert df[''] <  5

# Convert all > 5 to 5

df.loc[df[''] > 5, ''] = 5 row,column
```
**datetime:**
`df[''] = pd.to_datetime(df['']).dt.date`

today date: `today_date = dt.date_today()`

drop
or hardcode date with upper limit

### Unique Constraints

**Duplicate Values:**

**Parameter for `.duplicated`:**
- `subset`: List of columns to check
- `keep`: to keep firt, last, or all(False) duplicated values
- 
```python
duplicates = df.duplicated(subset =, keep =,)
df[duplicates]
```

**Parameter for `.drop_duplicates()`:**
- `subset`: List of columns to check
- `keep`: to keep first, last, or all(False) duplicated values
- `inplace`
```python
duplicates = df.drop_duplicates(subset =, keep =, inplace = )
df[duplicates]
```
---

```python
# Group by column names, produce stat summaries

column_names = ['','','']
summaries = {'': 'fn', '':'fn'}
df = df.groupby(by = column_names).agg(summaries).reset_index()
```
## Text and Categorical Problems

### Membership Constraints

- Dropping Data, Remapping, Inferring Categories

**Finding inconsistent categories:**
```python
inconsistent_categories = set(df['']).difference(categories['blood_type'])

inconsistent_rows = study_data['blood_type'].isin(inconsistent_categories)
df[inconsistent_rows]
```

### Categorical Variables

- Value inconsistency
- too many categories to few
- type

**Value inconsistency:**
`str.upper` / `str.lower`
`str.strip()`

**Collapsing Categories:**
```python
group_names = []
range = [, , np.inf] -> #bins arguement
df[] = pd.qcut(df[], q = 3, labels = group_names, bins = ranges)


## Many to Few
mapping = {'Current': 'Desired', 'Current': 'Desired'}
df[] = df[].replace(mapping)
df[].unique()
```
### Cleaning Text Data

- Input Validation using assert e.g. phone number length

Only want int:
`df[] = df[].str.replace(r'\D+', '')`

## Advanced Data Problems

### Uniformity
 Units
 Find wrong units then convert

DateTime Errors:

```python
df[] = pd.to_datetime(df[],
                      infer_datetime_formate = True,
                      errors = 'coerce')
```

e.g. 2019-03-08
- convert to NA
- infer format by understanding data source
- infer format from previous data

### Cross Field Validation

use multiple fields to check integrity of dataset
```python
formula = df[['','']].sum(axis = 1)
rows_ = formula == df[]

inconsistent_values = df[~rows_]
```

### Completeness

Missing Data:

```python
import missingno as msno

msno.matrix(df)
plt.show()

missing = df[df[''].isna()]
complete = df[~df[''].isna()]

## sort then plot again to see if random 

```
Missingness type:
- Missing Completely at Random 
- Missing at Random - relationship between other observed values
- Not Missing at Random - not in data
  
## Record Linkage

### Comparing Strings

Min edit distance -> number of edits

```python
from thefuzz import fuzz

fuzz.WRatio('','') -> 0 to 100. 100 exact match
# Good for partial string comparison

from thefuzz import process

string = ''
choices = pd.Series(['','','',''])

process.extract(string, choices, limit = 2)
```

Remapping using string matching

```python
categories[''] #series on correct categories

for state in categories['']:
    matches = process.extract(state, survey[''], limit = survey.shape[0])
    for potential_matches in matches:
        if potential_matches[1] >= 80:
            df.loc[df[''] == potential_match[0], 'state'] == state

```

### Generating Pairs

Record linkage: generate pairs from different df
compare pairs, score pairs, link pairs

blocking: generate pairs based on matching column

```python
import recordlinkage

indexer = recordlinkage.Index()

indexer.block('') ## Blocking
pairs = indexer.index(df1, df2)

compare_cl = recordlinkage.Compare()

compare_cl.exact('','', label = '') -> compare column from df1 to df2

compare_cl.string('', '', threshold = , label = '')

potential_matches = comapre_cl.compute(pairs, df1, df2)

potential_matches[potential_matches.sum(axis = 1) => 2]
```

### Linking df

`potential_matches[potential_matches.sum(axis = 1) => 3]`
shows matches in rows with matches in 3 columns between df1 and df2

`duplicate_rows =  matches.index.get_level_values(1)`

Find new rows
`df2_new = df2[~df.index.isin(duplicate_rows)]`

`full census = df1.append(df2_new)`
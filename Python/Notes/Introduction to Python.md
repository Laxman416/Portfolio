# Introduction to Python 

Course on Datacamp - 'Introduction to Python' and 'Intermediate Python'

- [Introduction to Python](#introduction-to-python)
  - [Numpy](#numpy)
    - [Introduction](#introduction)
    - [2D Numpy Array](#2d-numpy-array)
    - [Basic Statistics](#basic-statistics)
- [Intermediate Python](#intermediate-python)
  - [Matplotlib](#matplotlib)
    - [Data Visualisations](#data-visualisations)
    - [Customisation](#customisation)
  - [Dictionaries \& Pandas](#dictionaries--pandas)
    - [Dictionaries](#dictionaries)
  - [Pandas](#pandas)
    - [Index and Select Data](#index-and-select-data)
    - [Filtering on Pandas](#filtering-on-pandas)
  - [Loops](#loops)
    - [Iterating Dictionaries and Numpy](#iterating-dictionaries-and-numpy)
    - [Iterating Pandas](#iterating-pandas)
  - [Random Numbers](#random-numbers)


## Numpy

### Introduction

List Recap:
- Collection of values - can hold different type
  
Numpy list is better:
- Mathematical operations over collections
- Speed
- 

```python
import numpy as np

height = np.array(height)
weight = np.array(weight)

bmi = np.weight / np_height **2
```

NumPy arrays only contain one type

### 2D Numpy Array

`type(np_array) -> numpy.ndarray`
Can create any dimensional arary

```python
import numpy as np

np_2d = np.array[[1,2,3],[1,2,3]]
```

**Attribute**:
  `np_2d.shape -> (2,3) 2 rows, 5 columns`
  `np_2d[0] -> array([1,2,3])`
  `np_2d[0][2] -> row 0 column 3`

How to information of 2nd and 3rd family member
  `np_2d[:, 1:3]` -> all rows, 2nd and 3rd column

### Basic Statistics

**Functions:**
- `np.mean(array)`
- `np.median(array)`
- `np.corrcoef()`
- `np.std()`
- `np.sum(array1, array2)`
- `np.sort(array)` ->> could use parameter `order` when working with structured arraysS
- `np.argsort(array[:, 0])`
- `np.logical_and()`
- `np.logical_or()`
- `np.logical_not()`

# Intermediate Python

- Visualisations: matplotlib
- Data Structure: dictionaries and pandas
- Control Structures: Loops
- Case Study

## Matplotlib

### Data Visualisations

```python
import matplotlib.pyplot as plt

plt.plot(x_axis_data, y_axis_data) ## Line
plt.show()

plt.scatter(x_axis_data, y_axis_data) ## Scatter Plot
plt.show()

help(plt.hist)

plt.hist(values, bins = int)
plt.show()
```
### Customisation

**Functions:**
- plt.x_axis('')
- plt.y_axis('')
- plt.title('')
- plt.yticks([float/int],[string if needed])
- plt.x_axis('')
- plt.x_axis('')
- plt.clf() -> cleans plot
## Dictionaries & Pandas

### Dictionaries

dictionary = {key:value, key2:value2}
Keys have to be 'immutable' objects

`dictionary[key] -> value` 

**Functions:**
- add: `dictionary['key3'] = value3`
- del: `del(dictionary[key3])`

## Pandas

```python
import pandas as pd

df = pd.DataFrame(dict) 

df.index = [index1,...]

df = pd.read_csv("path", index_col = 0) # read from csv and select column 1 as index
```

### Index and Select Data

- Square Brackets
- loc and iloc
  
**Square Brackets:**

- **Column Access**
`df[column_label]` -> returns array like object
`df[[column_label]]` -> returns dataframe

- **Row Access**
`df[1:4]` -> returns row 2,3 and 4

**loc (label based):** 
`df.loc[index1, index2, ...]` -> returns rows specified

`df.loc[[index1, index2],[column1, column2]]` -> returns row and columns specified

`df.loc[:,[column1, column2]]` -> returns all rows and columns specified

**iloc (integer position-based):** 

`df.iloc[[1]]` -> returns row 2
`df.iloc[[1,2][0,1]]` -> returns row 2,3 and columns 1 2

### Filtering on Pandas

`filtered_df = df[df["area"] > 8]` -> filters df 
`filtered_df = df[np.logical_and(df["area"] > 8, df["area"] < 10)]`

## Loops

### Iterating Dictionaries and Numpy

```python
for key, value in dictionary.items():
  print(value)
```

```python
for value in np.nditer(array):
  print(value)
```
### Iterating Pandas

```python
for label, row in df.iterrows():
  print(lab) 
  print(row) - panda_series

  print(f"{lab}: {row["capital"]}") # Selective Print
```
**Adding Columns**

```python
for label, row in df.iterrows():
  # Creates new Column
  df.loc[label, "name_length"] = len(row["country"])
```

**Apply**

`df["new_column_name"] = df["old_column].apply(fn_name)`

## Random Numbers

`np.random.randint(0,2)`-> Randomly generate 0 or 1
# Introduction to Python 

**Course on Datacamp:**<br> 
`'Introduction to Python'`<br> 
`'Intermediate Python'`<br> 
`'Python Toolbox'`<br> 
`'Writing Functions in Python'`<br> 

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
- [Python Toolbox](#python-toolbox)
  - [Using Iterators](#using-iterators)
    - [Introduction](#introduction-1)
    - [Enumerate and Zip](#enumerate-and-zip)
    - [Processing Large Data](#processing-large-data)
  - [List Comprehensions and Generators](#list-comprehensions-and-generators)
    - [List Comprehensions](#list-comprehensions)
    - [Advanced Comprehensions](#advanced-comprehensions)
    - [Generator](#generator)
- [Functions II - Decorators](#functions-ii---decorators)
  - [Docstrings](#docstrings)
  - [Context Managers](#context-managers)
  - [Decorators](#decorators)


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

`np.random.randint(0,2)`-> Randomly generate 0 or 

# Python Toolbox

## Using Iterators

### Introduction

iterable: list, strings, dictionaries
- has an iter() method

iterator:
- produces next value()

```sql
it = iter('word')
next(it) -> w
next(it) -> o ...

print(*it) -> splat operator, need to redefine

for key, value in dic.items(): -> unpack using .items()
    ...
```

for files iterates over lines

### Enumerate and Zip  

- **Enumerate:**
- takes iterable as argument
  
**Parameters:**
- `start = int` -> starts looping at that value
```python
e = enumerate(list) -> <class 'enumerate'>
e_list = list(e) -> [(0,value1),(0, value2)]
for index, value in enumerate(list):
    ...
```

- **Zip:**

```python
zip_iterator_of_tuples = zip(list1,list2)
[(a_l1,a_l2),(b_l1,b_l2)]
```

### Processing Large Data

load data in chunks 
perform in each chunk
discard chunk
load next chunk

pandas:
- `chunksize` in `pd.read_csv('', chunksize = )`

```python
for chunk in pd.read_csv('', chunksize = int):
    result.append(sum(chunk['x']))
total = sum(result)
```

## List Comprehensions and Generators

### List Comprehensions

for loops inefficient

```python
nums = [...]
new_list = [num + 1 for num in nums]
```

collapses for loops and requirements:
- iterable
- iterator variable
- output expression

nested list comprehensions
`list2 = [(num1, num2) for num1 in range(0,2) for num2 in range(6,8)]`

### Advanced Comprehensions

- conditionals

`[num ** 2 for num in range(10) if num % 2 == 0]`
`[num ** 2  if num % 2 == 0] else 0 for num in range(10)`

- Dictionary:

use {} instead of []
key: value
``
### Generator

[] replaced with ()
`(num ** 2 for num in range(10) if num % 2 == 0)`

generators:
- doesnt store list in memory, but iterable list
- writes the code but doesnt generate memory

**Generator Fn:**
- produce generator objects
- `yield` instead of `return`

# Functions II - Decorators

## Docstrings

"""
Description: What fn does

Args:

Returns:


"""

`/__.doc__`
`inspect.getdoc()`
**Dont Repeat and Do One Thing**

pass by assignment

integers are immutable - cant be changed by fn

## Context Managers
Set up context
Run code 
Remove context

```python
with open() -> #context manager
```

**Writing Context Managers:**
- Class-based
- Function-based

```python
@contextlib.contextmanager
def fn():
    # code need
    yield
    # teardown code

```

**Advanced Topics:**

*Nested contexts:*

```python
@contextlib.contextmanager
def copy(src, dst):

    with open(src) as f:
        with open(dst, 'w') as f_dst:
            for line in f:
                f_dst.write(line)
    
    try:
        yield
    finally:
        # tear down code
```

## Decorators

Fn another object
Global and local scope
non local is in parent fn used in child fn
closures: fn contains tuple of memory
- non local variables in parent fn to returned when calling child fn

Wrapper around fn

```python
@double_args
def multiply(a,b)
    return a * b
# Decorator will double args -> 4* fn

def double_args(func):
    def wrapper(a,b):

        return func(a * 2, b * 2)

    return wrapper
 
multiply = double_args(multiply) #don't need use decorator syntax
multiply(1,5) -> # 20
```

```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)

        t_total = time.time() - t_start
        print(t_total)
        return reult
    return wrapper
```

```python
def memoize(func):
    # Store results in dict that maps arguments to results, doesnt create new as always in closure
    cache = {}

    @wraps(func) -> # modify wrapper metadeta to look like calling fn meta data
    def wrapper(*args, **kwargs):
        # key for kwargs
        kwargs_key = tuple(sorted(kwargs.items()))
        if (args, kwargs_key) not in cache:
            # store result
            cache[(args, kwargs_key)] = func(*args, **kwargs)
        return cache[(args, kwargs_key)]
    return wrapper
```
When to use:
- common code to all fn

Metadata:
use `from functools import wraps`

easy access to undecorated fn using `__.wrapped__`

**Decorators that take arguments**

```python
def run_n_times(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator
    
@run_n_times(3)
def print_sum(a,b)
    print(a+b)

```

**Timeout()**

```python
import signal

def timeout_in_5s(n_seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.alarm(5)
        try:
            # raise Timeout Error if too long
            return func(*args, **kwargs)
        finally:
            # cancel alarm
            signal.alarm(0)
        return wrapper
    return decorator


```
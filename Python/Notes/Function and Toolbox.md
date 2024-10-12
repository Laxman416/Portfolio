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


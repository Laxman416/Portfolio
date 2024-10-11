# Data Manipulation with pandas 

Course on Datacamp - 'Data Manipulation with pandas' and 'Joining Data with Pandas'

## Transforming DataFrames

### Introducing DataFrames

pandas uses matplotlib and numpy

Exploring a DF:
- `print(df.head())` -> Prints first few rows
- `print(df.info())` -> Display names of columns, and data types
- `print(df.shape)` -> (#rows, #columns)
- `print(df.describe())` -> shows summary statistics for columns like mean and medium.

Attributes:
- `df.values` -> data values
- `df.columns` -> column names
- `df.index` -> row/index names

### Sorting and Subsetting

Fn:
- `df.sort_values("column_name", ascending = False)`
- `df.sort_values([]"column_name", column2],ascending = [True, False])`
- `df["column_name"]` -> Subsetting column
- `df[["column_name","column2"]]` -> Subsetting 2 columns - 2 []
- `df[["column1"] > 50]` -> Subsetting Rows using condition


Use `&` or `||` to subset based on multiple conditions

Subsetting using .isin()

`df["column1].isin(["Black","Blue"])`

### New Columns/ Transforming DF

df[new_column] = operation * df["old_column"]

## Aggregating DF

### Summary Statistics

`df.[column_name].mean()`
`df.[column_name].median()`
`df.[column_name].min()`
`df.[column_name].max()`
`df.[column_name].mode()`
`df.[column_name].var()`
`df.[column_name].std()`
`df.[column_name].cumsum()`
`df.[column_name].cummax()`
`df.[column_name].cummin()`
`df.[column_name].cumprod()`

**.agg()**
```python
def pct30(column):
    return column.quantile(0.3)

df[column_name, column2].agg([pct30, pct40])
```

### Counting

```python
unique_df = df.drop_duplicates(subset = ["column_name", "breed"])

unique_df["column"].value_counts(sort=True, normalize = True)
```

### Grouped summary statistics

Can groupby and aggregate by multiple columns same time

`df.groupby("column")["column2"].mean()`
`df.groupby(["column", "column2"])["column2"].agg([min,max,sum])`

### Pivot Table

Group by to pivot table
Default mean
```python
df.pivot_table(values="column1",index="column2",columns="2ndgroupby column", aggfunc=[np.mean,np.median], fill_value = 0, margins=True)

```

parameter: margins = True -> gives mean of values

## Slicing and Indexing DataFrames

### Explicit Indexes

```python

df_ind = df.set_index(column) # Set Index

df_ind.reset_index(drop=True)

```
can create multi level index
subset outer level with list
subset inner levels with list of tuples

.sort_index(level = column_name)

### Slicing and Subsetting with .loc and .iloc

`df.loc["index1":"index2"]` -> final value included only on outer index levels


`df.loc[(tuple):(tuple)]` -> multilevel index

Slicing on columns:
- `df.loc[:, "column1":"column2"]` -> colon for all rows

index date then sort to do this ->
can splice by partial dates

### Working with Pivot Tables

`df.loc[row1:row2]` -> .loc() + slicing
`df.mean(axis='index')` -> calculated for each column
`df.mean(axis='columns')` -> calculated for each row

## Creating and Visualising DF

### Visualise your Data

```python
import matplotlib.pyplot as plt

# Hist
df[column1].hist(bins=int)
df[df["type"] == "organic"]["avg_price"].hist()
plt.show()

# Bar Chart
avg_of_column_by_group.plot(kind = 'bar',title=...)
plt.show()

# Line Chart
df.plot(x=column1, y=column2, kind='line')

# Scatter
df.plot(x=column1, y=column2, kind='scatter')

# Layered Ploted
```
### Missing Values
`df.isna().sum()` -> shows columns wiht NAN /.any()
can plot NANs 
`df.isna().sum().plot(kind='bar')` 

`df.dropna()` -> remove rows
`df.fillna(0)` -> if cant remove use fillna with staistical methods

### Creating DF
Two ways:
- from list of dictionaries -> row by row
- from dictionary of lists -> column by column

```python
dict = [
    {row1}.
    {row2}
]
```

```python
dict_of_lists = {
    column1_name : [value1,value2],
    column2_name : [value1,value2]

}

```
### Reading and Writing CSV

```python

df = pd.read_csv("file_path")

df.to_csv("new_file_path")
```

# Joining Data with Pandas

## Data Merging Basics

### Inner Join

- 'Inner Join' - joins matching values on both
- Suffixes to differentiate between matching column names between the two DF
- Number of rows fixed
  
```python
new_df = df1.merge(df2, on = 'matching_column', suffixes = ('_df1','_df2'))

```
### One to Many Relationship
One to One: Every Row in Left DF related to one row in right DF
One to Many: Every Row in LEFT related to many row in right DF

The LHS will be reated for every relationship

One to Many -> Increases # of rows

### Merging multiple DF

Single merge  -> on = [multiple_columns]

```python
df1_df2_df3 = df1.merge(df2, on = [multiple_columns]) \
    .merge(df3, on = 'column', suffixes = [])

```

## Merging DF with Different Join Types

### Left JOIN

All rows from left + matching rows from right if any
others on right kept as null
- keyword: `how`
  
```python

df1_df2 = df1.merge(df2. on = 'column', how = 'left')

```
### Other JOIN

Right JOIN similiar to LEFT JOIN
- keyword: `left_on` and `right_on` when column name not matching
```python

df1_df2 = df1.merge(df2. on = 'column', how = 'left', left_on = 'id', right_on = 'df2_id')

```

**Outer JOIN**
Full JOIN - all rows returned

`how = 'outer'`
### Self JOIN
- Inner JOIN
can also do left JOIN if specified with how

```python
new_df1 = df1.merge(df1, left_on = '', right_on = '', suffixes = '')

```

### Merging on Indexes

multiIndex
```python

new_df = df1.merge(df2, on = ['id_1','id_2'])
```

Index merge with left_on and right_on
- Keyword: `left_index = True` and `right_index = True`
```python

new_df = df1.merge(df2, left_on = 'id_1', left_index = 'True', right_on ='id_2', right_index = True)
```

## Advanced Merging and Concatenating

### Filtering JOINS

- Filter observations from DF based on whether or not they match an observation in another DF
- SEMI JOIN:
  - similiar to inner join, returns only left table


```python
df3 = df1.merge(df2, on = 'column')
new_df = df1[df1['column'].isin(df3)]
```

Anti-JOIN:
Returns left table which do not match with right table

```python
df3 = df1.merge(df2, on = 'column', how = 'left', indicator = True)
column_list = df1.loc[df3[-_merge] == 'left_only', 'column1']
df4 = df1[df1['column'].isin(column_list)]
```
### Concatenate DF Vertically

`.concat()`: axis=0 is vertical

`pd.concat([df1,df2,df3], ignore_index = True), keys = [], sort = True/ join = 'inner'`: keys are labels

### Verifying integrity

Unintentional one to many / many to many

.merge(validate='one_to_one')

other options for validate:
  - `one_to_one`
  - `one_to_many`
  - `many_to_one`
  - `many_to_many`
  
Verifying Concat:
`.concat(verify_integrity = True)`
Checks wheather index values have duplicates

## Merging Ordered and Time-Series Data

### Using merge_ordered()

sorted merge

`pd.merge_ordered(df1,df2)`:
- Columns to join on: `on, left_on, right_on`
- Type of join: `how (left, right, inner, outer)
- default is outer
- supports suffixes


**Forward Fill**

fills missing values from previous values

`fill_method = "ffill"`
  
### Using merge_asof()

Similar to `merge_ordered()`

Match on nearest key column and not nearest matches
- Merged on columns must be sorted
- Merged on columns on RHS will be matched to nearest values lower than or equal to
- `parameter: 'directon' = 'forward'` will do values nearest from the > than side.
- `direction = nearest`: direction doesnt matter, just closest
  
### Selecting data with .query()

.query('WHERE STATEMENT')

`df1.query('column > 90')`

can use .query to select data

### Reshaping data with .melt()

Wide Format to Long Format can change between them

Long: first last variable value
Wide: first last height weight

```python
df_tall = df.melt(id_vars = ['column1', 'column2'])
```
- id_vars: columns you dont want to melt
- value_vars =` ['variable','variable '] ` selects which variables you want to show
- change title: `var_name = ''`, `value_name = ''` 
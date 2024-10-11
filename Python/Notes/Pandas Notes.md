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
# Categorical Data
Course on DataCamp
- pandas and seaborn
## Introduction to Categorical Data

Categorical:
- Finite, Qualitative

Ordinal if natural order, Nominal if doesn't have natural order

`.info()`
`.describe()`
`.value_counts(normalize = True)` - on panda series

### Categorical data in pandas

`df.dtypes` 
object -> Categorical

`df[''] = df[''].astype("category")`

creating a categorical series
```python
series = pd.Series(my_data, dtype='category')
series = pd.Categorical(my_data, categories=['','',''], ordered = True)
```

categorical dtype memory < object

```python
pd_dtypes = {'':'category'}
df = pd.read_csv('', dtype= pd_dtype)
```

### Grouping Data by Category Pandas

`.groupby()` -> splits data
`.size()` -> counts of groupby

## Categorical Panda Series

### Setting Category Variables

`.cat` accessor object

`Series.cat.method_name`

**Common parameters:**
- new_categories: a list 
- inplace: Boolean to overwrite
- ordered: Boolean: if ordered

**Method_Names:**
- `cat.set_categories`
- `cat.add_categories`

### Updating Categories

**Method_Names:**
- `rename_categories(new_categories=dict)`
renaming with lambda fn
`df[column] = df[column].cat.rename_categories(lambda c: c.title())`
new category must not be new
- `.replace(dict/list)`
need to convert back to to categorical

### Reordering Categories

**Method Name:**
`reorder_categories(ordered = True, inplace = True)`

### Cleaning and Accessing Data

spelling issues/ inconsistent values

`df[] = df[].str.strip()` -> removes white space
`df[] = df[].tittle()` -> fixing capitalisation

```python
replace_map = {'':''}
df[].replace(replace_map, inplace = True)

df[].str.contains('', regex = False) -> #treted as a literal string

```

Accessing Categories

```python

df.loc[df[column], "category"]

```
Series.cat.categories
series.value_counts()

## Visualising Categorical Series

### Boxplots

`sns.catplot()`

**parameters:**
- `x`
- `y`
- `data`
- `kind`: strip, swarm, box, violin, boxen, point, bar, count

`sns.set_style('whitegrid')`

### Bar Charts

`ci`
`order`
`hue`

### Point Plot

`dodge = True, join = True`

**Additional catplot options:**

facetgrid

`col = column` -> plot for each column

## Pitfalls and Encoding

### Categorical Pitfalls

Memory proportional to num of categories

using `.str` or `.apply()` -> object series
NumPy functions dont work

- Check befores setting categories

### Label Encoding

Codes category as integers

`df[] = df[].cat.codes` -> creates index like codes for categories

```python
codes = df[].cat.codes
categories = df[]

name_map = dict(zip(codes,categories)) -> dictionary

df[].map(name_map)
```

Boolean Coding

```python

df[] = np.where(df[].str.contains("", regex= False), 1,0)
# Set to 1 or 0 depending if it contains the string

```

### One-hot encoding

`pd.get_dummies()`

**parameters:**
- data
- columns
- prefix

creates new columns for each categorical columns


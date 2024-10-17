# Exploratory Data Analysis in Python

'course from Datacamp'

- [Exploratory Data Analysis in Python](#exploratory-data-analysis-in-python)
  - [Introduction](#introduction)
    - [Exploration](#exploration)
    - [Data Validation](#data-validation)
    - [Data summarisation](#data-summarisation)
  - [Data Cleaning and Imputation](#data-cleaning-and-imputation)
    - [Missing Data](#missing-data)
    - [Converting and analysing categorical data](#converting-and-analysing-categorical-data)
    - [Numeric Data](#numeric-data)
    - [Handling outliers](#handling-outliers)
  - [Turning Exploratory Analysis into Action](#turning-exploratory-analysis-into-action)
    - [Patterns over time](#patterns-over-time)
    - [Correlation](#correlation)
    - [Categorical Relationships](#categorical-relationships)
  - [EDA](#eda)
      - [Considerations for Categorical Data](#considerations-for-categorical-data)
    - [Generating new features](#generating-new-features)
    - [Generating Hypotheses](#generating-hypotheses)


## Introduction

### Exploration

EDA: process of reviewing and cleaning data:
- derive insights
- generate hypotheses

`df.value_counts()`
`df.describe()`
`df.info()` -> datatypes and non missing values count

### Data Validation

`df.dtype()`

Changing Data type of columns`df[column] = df[column].astype(int)`

Validating categorical data:

`df[column].isin(['',''])` True or False
`~df[column].isin(['',''])` False or True
`df[df[column].isin(['','']])` -> filtering using `.isin()` 

`df[column].min()` -> outputs value
`df[column].max()` -> outputs value

### Data summarisation

`.groupby()` -> groups data
`.groupby(column).agg(['mean', std])` -> agg fn how to sumarise grouped data

**Agg fn:**
- df.agg({column1: functions, column2:function2})

`df.groupby(column).agg(mean_rating = ('','mean'), std_rating = ('','std'))` -> named summary columns

barplots to visualise categorical summaries

## Data Cleaning and Imputation

### Missing Data

Affects distributions:
- Missing heights of taller students
Less Representatives of the population
-> incorrect conclusions
  
Check for missing values - `df.isna().sum()`

Solutions:
- Drop: using conditions, if na > 0.05 dataset remove column
```python

cols_to_drop = df.columns[df.isna().sum() <= threshold]
df.dropna(subset = cols_to_drop, inplace = True)
```
- Impute mean,median or mode
```python
cols_with_missing_values = df.columns[df.isna().sum() >0 threshold]

for col in cols_with_missing_values[:-1]:
    salaries[col].fillna(df[col].mode()[0])
```

Check for missing values again:
imputing by sub-group

```python
df_dict = df.groupby('column')['column'].median.to_dict()
df[column] = df[column].fillna(df[experience].map(df_dict))
```

```python
df.fillna(method = 'ffill') -> #fills last value for NaN
```
### Converting and analysing categorical data

Extracting value from categories - `df[''].str.contains('Scientist|AI')`

Starting with phrase: `df[''].str.contains('^Data')`

```python
conditions = [(df[''].str.contains('Data'),())]
df[new_column] = np.select(conditions, job_categories, default = 'Other')
```
### Numeric Data

Converting strings to numbers:
- remove comas
`df[] = df[].str.replace(",", "")`

`df[] = df[].astype(float)`

Adding summary statistics into DF
```python 
df["std_dev"] = df.groupby("")[""].transform(lambda x: x.std())

```

### Handling outliers

outlier is when lies outside of range 25PERCENTILE + 1.5*IQR

## Turning Exploratory Analysis into Action


### Patterns over time

```python
df = pd.read_csv('', parse_dates = [column])
df[column] = pd.to_datetime(df[column])

#Creating DT data

df[column] = pd.to_datetime(df[['month','day','year']]])

df['month'] = df[''].dt.month # Extracts month

```
### Correlation

`df.corr()`
`sns.heatmap(df.corr(), annot = True)`

`sns.pairplot(data = df, vars = [variables of interest, '', ''])`

### Categorical Relationships

- Hist are useful, hut kde are better - smoothing of hist

`sns.kdeplot(data, x, hue, cut = 0)` -> cut forces smoothing between min and max

**parameter:**
- `cut`:
- `cumulative`


## EDA

#### Considerations for Categorical Data

- Detecting patterns
- Generating hypothesis
- preparing data for machine learning

Sample must represent data
Categorical classes:
- classes = labels
- Marriage status: marriage, single, married
- Class imbalance: 
`planes[''].value_counts(normalize = True)`

Cross-tabulations:

pd.crosstab(planes['Source'], planes['Destination'])
-> grid


pd.crosstab(planes['Source'], planes['Destination'], values = planes['Price'], aggfunc = 'median')

-> shows median values for all possible

compare median price of dataset and compare with real data, to see if class imbalances

### Generating new features

e.g.: convert DT into month weekday columns

Creating categories:
Split into quantile
create label 
create bin width = [0,...,1]
`pd.cut(df[''], labels = labels, bins = bins)`

### Generating Hypotheses


# Data Analysis and Exploration <!-- omit in toc -->

**Course on Datacamp:**<br> 
`'Introduction to Statistics'`<br> 
`'Exploratory Data Analysis in Python'`<br> 
`'Categorical Data'`<br> 

- [Introduction to Statistics](#introduction-to-statistics)
  - [Summary Statistics](#summary-statistics)
    - [Statistics](#statistics)
    - [Measures of Centre](#measures-of-centre)
    - [Measures of Spread](#measures-of-spread)
  - [Random Numbers and Probability](#random-numbers-and-probability)
    - [Probability](#probability)
    - [Discrete Distributions](#discrete-distributions)
    - [Continuous Distributions](#continuous-distributions)
    - [The Binomial Distributions](#the-binomial-distributions)
  - [More Distributions and CLT](#more-distributions-and-clt)
    - [The Normal Distribution](#the-normal-distribution)
    - [The Central Limit Theorem](#the-central-limit-theorem)
    - [Poison Distributions](#poison-distributions)
    - [Other Distributions](#other-distributions)
  - [Corelation and Experimental Design](#corelation-and-experimental-design)
    - [Correlation](#correlation)
    - [Correlation Caveats](#correlation-caveats)
    - [Design of Experiments](#design-of-experiments)
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
    - [Correlation](#correlation-1)
    - [Categorical Relationships](#categorical-relationships)
  - [EDA](#eda)
      - [Considerations for Categorical Data](#considerations-for-categorical-data)
    - [Generating new features](#generating-new-features)
    - [Generating Hypotheses](#generating-hypotheses)
- [Categorical Data](#categorical-data)
  - [Introduction to Categorical Data](#introduction-to-categorical-data)
    - [Categorical data in pandas](#categorical-data-in-pandas)
    - [Grouping Data by Category Pandas](#grouping-data-by-category-pandas)
  - [Categorical Panda Series](#categorical-panda-series)
    - [Setting Category Variables](#setting-category-variables)
    - [Updating Categories](#updating-categories)
    - [Reordering Categories](#reordering-categories)
    - [Cleaning and Accessing Data](#cleaning-and-accessing-data)
  - [Visualising Categorical Series](#visualising-categorical-series)
    - [Boxplots](#boxplots)
    - [Bar Charts](#bar-charts)
    - [Point Plot](#point-plot)
  - [Pitfalls and Encoding](#pitfalls-and-encoding)
    - [Categorical Pitfalls](#categorical-pitfalls)
    - [Label Encoding](#label-encoding)
    - [One-hot encoding](#one-hot-encoding)


# Introduction to Statistics 

## Summary Statistics

`.cdf()`prob of <= value
`.rvs` generate values from dist.
`.pmf()` -> prob of single value
### Statistics

Descriptive Statistics - Describe and summaries Data
Inferential Statistics - Sample to infer about large population

Two Types of Data:
- Quantitative: Continuous and Discrete
- Qualitative: Nominal / Ordinal -> Ordered
- 
### Measures of Centre

3 different measures of centre:
- `np.mean()`
- `np.median()`, or sort then find middle index in a DF 
- `statistics.mode()`

Outlier: mean more sensitive to extreme values and better for symmetric data.

Left-skewed and Right-Skewed: Mean pulled in direction of skew relative to median

### Measures of Spread

- **Variance**:
avg dist from data point to mean squared
`np.var(value , ddof=1)`
- **Std**:
sqrt of variance
`np.std(value , ddof=1)`

-> `ddof=1` is required when working with a sample

- **Mean absolute deviation**:
- std sum squares, while MAD doesnt
- std penialises longer distances more 

- **Quantitle**

`np.quantile(list, [0.25,0.5,0.75])`
`plt.boxplot()`

**IQR**
```python
from scipy.stats import iqr
iqr(list)
```
Outliers when 
**data < Q1 -1.5IQR**
OR 
**data > Q3 + 1.5IQR**

`.describe` -> **shows all summary stats in one line**

## Random Numbers and Probability

### Probability

`df.sample(num_of_rows, replace = True)` -> random row from DF

np.random.seed(10)

Independent events: first event doesn't change second prob.
Dependent events: first event changes second prob.

### Discrete Distributions

for discrete variables: histograms

Law of large numbers

### Continuous Distributions

Probability: Area on probability graph

```python
from scipy.stats import uniform

uniform.cdf(8,lower_limt,upper_limit) -> 'probabilty of 8 or less'

uniform.rvs(0,5, size=10) -> 'generates random values'
```

### The Binomial Distributions

```python
from scipy.stats import binom

binom.rvs(size = num_of_item, prob_success(p), how_many_times(n))

binom.pmf(value,n,p) -> gives prob(x = value) 
```

Expected value = n * p

Each trial must be independent

## More Distributions and CLT

### The Normal Distribution

normalised and symmetrical, prob never -> 0
standard normalised dist.: mean = 0, std = 1

```python
from scipy.stats import norm

norm.cdf(value, mean, std) -> prob(x <= value)

norm.rvs(mean, std, num_generated) -> generates rand numbers

pct_25 = norm.ppf(percentile,mean,std) -> prob. value

```

### The Central Limit Theorem

as # of trials increases, closer to normal dist.

- only when samples independent and random

### Poison Distributions

Poisson process: certain rate(time) but random -> mean

lambda = average rate

Probabilty of single value: `poisson.pmf(value, lambda)`
prob(x <= value): `poisson.cdf(value, lambda)`

`poisson.rvs(8,10)`

### Other Distributions

- **Exponential**: **prob of time between poisson events**

e.g. prob of >1 day between poisson events
- continuous
  
```python
from scipy.stats import expon

expon.cdf(value, scale = 1/lambda)
```
**t-dist.**
similiar to normal but tails are thicker

parameter:
- `df`: dof: higher df decreases std and closer to std normal


**log-normal dist**

variable log. distributed

## Corelation and Experimental Design 

### Correlation

**Pearson product-moment correlation (r)**
Relationship between variables:
r = -1 to 1
magnitude -> strength
sign -> direction

```python
import seaborn as sns

# Set trend line
sns.lmplot(x,y, data = df, ci = None) #ci confindence internal margins set to None

```

correlation in pandas columns
```python
df['column1'].corr(df['column2'])
```

### Correlation Caveats

r only uses linear trends

- Non-linear relationships

transformations:

log transformation - `df['log_column1'] = np.log(df['column1'])`
sqrt(x) : 1/y

Correlation doesn't imply causation - confounding

### Design of Experiments

- Aim to answer Question
- Controlled group comparable

remove bias:
- randomised controlled trial
- placebo
- Double-blind prevents bias in analysis

assigned people aren't random therefore only association not causation
 
longitudinal study: long period of time - expensive
Cross section study: single snapshot of time

# Exploratory Data Analysis in Python

'course from Datacamp'

- [Introduction to Statistics](#introduction-to-statistics)
  - [Summary Statistics](#summary-statistics)
    - [Statistics](#statistics)
    - [Measures of Centre](#measures-of-centre)
    - [Measures of Spread](#measures-of-spread)
  - [Random Numbers and Probability](#random-numbers-and-probability)
    - [Probability](#probability)
    - [Discrete Distributions](#discrete-distributions)
    - [Continuous Distributions](#continuous-distributions)
    - [The Binomial Distributions](#the-binomial-distributions)
  - [More Distributions and CLT](#more-distributions-and-clt)
    - [The Normal Distribution](#the-normal-distribution)
    - [The Central Limit Theorem](#the-central-limit-theorem)
    - [Poison Distributions](#poison-distributions)
    - [Other Distributions](#other-distributions)
  - [Corelation and Experimental Design](#corelation-and-experimental-design)
    - [Correlation](#correlation)
    - [Correlation Caveats](#correlation-caveats)
    - [Design of Experiments](#design-of-experiments)
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
    - [Correlation](#correlation-1)
    - [Categorical Relationships](#categorical-relationships)
  - [EDA](#eda)
      - [Considerations for Categorical Data](#considerations-for-categorical-data)
    - [Generating new features](#generating-new-features)
    - [Generating Hypotheses](#generating-hypotheses)
- [Categorical Data](#categorical-data)
  - [Introduction to Categorical Data](#introduction-to-categorical-data)
    - [Categorical data in pandas](#categorical-data-in-pandas)
    - [Grouping Data by Category Pandas](#grouping-data-by-category-pandas)
  - [Categorical Panda Series](#categorical-panda-series)
    - [Setting Category Variables](#setting-category-variables)
    - [Updating Categories](#updating-categories)
    - [Reordering Categories](#reordering-categories)
    - [Cleaning and Accessing Data](#cleaning-and-accessing-data)
  - [Visualising Categorical Series](#visualising-categorical-series)
    - [Boxplots](#boxplots)
    - [Bar Charts](#bar-charts)
    - [Point Plot](#point-plot)
  - [Pitfalls and Encoding](#pitfalls-and-encoding)
    - [Categorical Pitfalls](#categorical-pitfalls)
    - [Label Encoding](#label-encoding)
    - [One-hot encoding](#one-hot-encoding)


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

# Categorical Data
Course on DataCamp
- pandas and seaborn

- [Categorical Data](#categorical-data)
  - [Introduction to Categorical Data](#introduction-to-categorical-data)
    - [Categorical data in pandas](#categorical-data-in-pandas)
    - [Grouping Data by Category Pandas](#grouping-data-by-category-pandas)
  - [Categorical Panda Series](#categorical-panda-series)
    - [Setting Category Variables](#setting-category-variables)
    - [Updating Categories](#updating-categories)
    - [Reordering Categories](#reordering-categories)
    - [Cleaning and Accessing Data](#cleaning-and-accessing-data)
  - [Visualising Categorical Series](#visualising-categorical-series)
    - [Boxplots](#boxplots)
    - [Bar Charts](#bar-charts)
    - [Point Plot](#point-plot)
  - [Pitfalls and Encoding](#pitfalls-and-encoding)
    - [Categorical Pitfalls](#categorical-pitfalls)
    - [Label Encoding](#label-encoding)
    - [One-hot encoding](#one-hot-encoding)

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


# Introduction to Statistics 

Course on Datacamp - 'Introduction to Statistics

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


#  Statistical Foundations: Modeling, Sampling, and Hypothesis Testing<!-- omit in toc -->

**Course on Datacamp:**<br> 
`'Introduction to Regression with statsmodels in Python'`<br> 
`'Sampling in Python'`  
`'Hypothesis Testing'`<br>
`'Experimental Design in Python'`<br> 

- [Regression with statsmodels](#regression-with-statsmodels)
  - [Simple Linear Regressions](#simple-linear-regressions)
  - [Predictions and model objects](#predictions-and-model-objects)
  - [Assessing model fit](#assessing-model-fit)
  - [Simple Logistic Regression Modelling](#simple-logistic-regression-modelling)
- [Sampling](#sampling)
  - [Sampling Fundamentals](#sampling-fundamentals)
  - [Sampling Methods](#sampling-methods)
  - [Sampling Distributions](#sampling-distributions)
  - [Bootstrap Distributions](#bootstrap-distributions)
- [Hypothesis Testing](#hypothesis-testing)
  - [Hypothesis Testing Fundamentals](#hypothesis-testing-fundamentals)
  - [Two-Sample and ANOVA Tests](#two-sample-and-anova-tests)
  - [Proportion Tests](#proportion-tests)
  - [Non-Parametric Tests](#non-parametric-tests)
- [Experimental Design in Python](#experimental-design-in-python)
  - [Experimental Design Preliminaries](#experimental-design-preliminaries)
  - [Experimental Design Techniques](#experimental-design-techniques)
  - [Analyzing Experiment Data: Statistical Tests and Power](#analyzing-experiment-data-statistical-tests-and-power)
  - [Advanced Insights from Experimental Complexity](#advanced-insights-from-experimental-complexity)

# Regression with statsmodels

## Simple Linear Regressions

In this subchapter:
- Two variables
- Fitting 
- Linear regressions with `ols()`
- Categorical explanatory variables

Regression: given explanatory(independent - x) variables predict value of response(dependant - y) variable

Good to visualise data using regplot
`sns.regplot(x = '', y = '', data = , ci = None)`

Packages:
`statsmodels` - optimised for insight
`scikit-learn` - optimised for predictions

**Fitting a linear regression:**
straight lines
```python
from statsmodels.formula.api import ols

mdlobject = ols("y ~ x", data = )
mdlobject.fit()
```
*Parameters for `ols('y ~ x', data = )`*</b>

`data` </b>

`"response variable ~ explanatory variable"` -> "y vs x" </b>

*Fn for model object:* </b>
- `mdlobject.fit()` </b>
- `mdlobject.params` -> contains params </b>

**Categorical explanatory variables**

`sns.displot(col)` -> multiple histograms
intercept value same -> mean mass
gradient individual for each and relative to intercept. intercept + gradient = mean

`ols('y ~ x + 0')`
-> gradient is now mean of each category

## Predictions and model objects

In this subchapter:
- Making Predictions
- Working with model objects 
- Regression to the mean
- Transforming Variables

Data on explanatory values to predict:

```python
explanatory_data = pd.DataFrame("": np.arrange(20,41))
explanatory_data.predict()

predicted_data = explanatory_data.assign(name_column = mdl_object.predict("explanatory_data"))

```

*Attributes of model object:*
- `mdl-object.predict()`

set matlpotlib fig to plot plots on top of another
`fig = plt.figure()`

Extrapolating

**Working with model objects**

Attributes/Fn for model object:
- `mdlo.params`
- `mdlo.fittedvalues`-> predictions on original dataset
- `mdlo.resid` residuals -> actual response values minus predicted
- `mdlo.summary()`

**Regression to the mean:**
- used to quantify the effect

response = fitted + residual

**Transforming Variables:**

transform x:
*log:*
*cubed*
*squared*
*sqrt* -> to spread data out if right skewed

need to transform explanatory data 
need to square predicted data -> back transformation

## Assessing model fit

In this subchapter:
- Quantifying model fit
- Visualising model fit
- Outliers, leverage, and influence
  
**Quantifying Model Fit**

- `r-squared:` proportion of variance from y that is predictable from x
1 is perfect fit

`mdlo.summary()`
`mdlo.r_squared`

- Residual standard error

typical error distance. can also use Mean Square Error

`mdlo.mse_resid`-> MSE </b>

`np.sqrt(mse)` -> RSE

sum of square of residuals / dof(observations - parameters)

- RMSE

sum of square of residuals / dof(observations)

**Visualising Data**

- **Residuals vs fitted** - trend line should follow y = 0
or could do count of residuals vs fitted

`sns.residplot(x =, y=, data = , lowess= True)`

- **Q-Q:** Theoretical Quantiles vs Sample Quantiles
wheather dist. follows normal
follow line -> good dist.
```python
from statsmodels.api import qqplot

qqplot(data = mdlo.resid, fit = True, line = '45')
```
- **Scale-location plot:**
sqrt(Standard Residuals) vs fitted
```python
model_norm_residuals = mdlo.get_influence().resid_studentized_internal
model_norm_residuals_sqrt = np.sqrt(np.abs(model_norm_residuals))

sns.regplot(x = mdlo.fittedvalues, y = model_norm_residuals_sqrt, ci =None, lowess = True)
```

**Outliers**

Extreme x values -> use leverage: how extreme values are
Extreme y values

Influence, how much model will change if removed -> 
```python
summary_frame = mdlo.get_influence().summary_frame() 
df['leverage'] = summary_frame["hat_diag]
df['Cooks Dist'] = summary_frame["cooks_d"]
```

Plot removed outliers and not removed fits on same graph

## Simple Logistic Regression Modelling

In this subchapter:
- Why you need Logistic Regression: When response variable is logical -> S curve
- Predictions and odd ratios
- Quantifying logistic regression fit

```python
from statsmodels.formula.api import logit

mdlo = logit('y ~ x', data = ).fit()
sns.regplot(logistic = True)
```

**Predictions and Odds ratios**
```python
explanatory_data =
predicted_data = explanatory_data.assign(df_name = mdlo.predict(explanatory_data))

# Most likely outcome when prob >0.5
predicted_data["most_likely"] = np.round(predicted_data[""])
```

odds_ratio = prob/1-prob
if < 1 most likely outcome swaps

log odds ratio -> change linearly
`plt.yscale('log')`

**Quantifying:**
false positives, false negatives

```python
outcomes = pd.DataFrame({"actual_response": actual_response,"predicted_response": predicted_response})
print(outcomes.value_counts(sort = False))


# Confusion Matrix
conf_matrix = mdlo.pred_table() 
# true negative    false positive
# false negative   true positive

from statsmodels.graphics.mosaicplot import mosaic
# Plots Confusion Matrix
mosaic(conf_matrix)
```

*Accuracy*: - TN + TP/ ALL # of correct</br>
*Sensitivity*: - TP/ FN + TP # proportion of true positive found</br>
*Specificity*: TN/TN + FP</br>

# Sampling

## Sampling Fundamentals

In this subchapter:
- Sampling and point estimates
- Convenience sampling
- Pseudo-random number generation

`df.sample(n = int, random_state =)` -> n is number of unique rows sampled
`df[''].sample(n =, random_state = )` random state is seed

population parameter: calculation made on dataset
point estimate: calculation made on sample

**Convenience sampling**

sample has to be representative, otherwise sample bias

can compare dist of sample using hist and compare with pop

```python
sample[].hist(bins = np.arange(59,93,2))
df_population[].hist(bins = np.arange(59,93,2))
```

**Pseudo-random**

from physical process - expensive
pseudo-random cheap and fast -> calcualted from previous number using seed.

`np.random.seed(seed_num)`
`numpy.random.fn_name` e.g. `.normal()`

## Sampling Methods

In this subchapter:
- Simple random and systematic sampling
- Stratified and weighted random sampling
- Cluster sampling
- Comparing sampling methods

**Simple random and systematic sampling**

simple random - randomly pick using `.sample`
systematic sampling - picking at specific intervals

Systematic Sampling
```python
sample_size = 5
pop_size = len(df)

interval = pop_size // sample_size

df.illoc[::interval] ## :: does systematic sampling
```
can introduce bias
``` python
df_shuffled = df.sample(frac = 1) #  Shuffles df 
df_shuff;ed = df_shuffled.reset_index(drop =True).reset_index() -> # clears previous index
```

**Stratified and weighted random sampling**

sampling subgroups  -> ensures sample has same dist of subgroups
```python
df_sample = df.groupby('').sample(frac = 0.1, random_state = int)

## Sampling with equal counts
df_sample = df.groupby('').sample(n = 10, random_state = int)


## Weighted Random Sampling
df_weight = df
condition = df_weight[''] == ''

df_weight["weight"] = np.where(condition, 2 , 1)
```

**Cluster Sampling**

Stratified:
- split pop into subgroups
- simple random on every subgroup
Cluster:
- Use simple random to pick some subgroups
- use simple random on those subgroups

```python
# cut down to pick some sib groups
df_group_sample = random.sample(df_pop, k =3) # groupnames
group_condition = df[column_of_group].isin(df_group_sample)

df_cluster = df[group_condition]
df_cluster[column_of_group] =  df[column_of_group].cat.remove_unused_categories()

df_sample = df_cluster.groupby(column_of_group).sample(n =5, random_state = int)

```
- Multistage sampling. Can have more than 2 stages

**Comparing sampling methods**

`df_sample.shape`

Simple Random: - close to sample mean
Stratified: Groups first before sampling - close to sample mean
Cluster: only selects some groups before sampling - less data

## Sampling Distributions

In this subchapter:
- Relative error of point estimates
- Creating a sample dist.
- Approximate sampling dist.
- Standard errors and CLT

**Relative error of point estimates**

$\text{rel\_error} = 100 \times \frac{\lvert \text{pop\_mean} - \text{sample\_mean} \rvert}{\text{pop\_mean}}$

Properties of rel error vs sample error
- noisy - for small
- amplitude steep then flattens
- rel_error -> 0 as sample_size -> infinity

**Creating a sample dist.**

store means of same sample size multiple times
plot.hist - CLT as number of times ran increases

Approximate sample dist:
`expand_grid({['die1':[1,2,3,4]],[]})`

add column for mean -> type to category and use barplot

```python
df['mean'].value_counts(sort = False).plot(kind = 'bar')
```

if posibitlies to large need to approximate

```python
for i in range(1000):
  sample_means.append(
    np.random.choice(list(range(1,7)), size = 4, replace = True).mean()
  )
```

**Standard errors and CLT**

average of independent samples -> normal dist.

as sample size increase:
- dist of averages closer to be normally dist.
- width of sampling dist. narrower

calculated std of pop use ddof = 0
for sample ddof = 1

$\text{std\_sample} = \text{std\_pop} / \sqrt\text{(sample\_size)}$

std_sample -> standard errors

## Bootstrap Distributions

In this subchapter:
- Comparing sampling and bootstrap
- Confidence intervals

sampling with replacement (resampling)

Bootstrapping:
building up a theoretical population from sample

- sample with replacement to same size of original sample `frac - 1, replace = True`
- calculate statistics like mean
- replicate

-> bootstrap distribution

Comparing:
Bootstrap mean close to sample mean.
If original sample not good estimate of pop mean.
Bootstrapping can't correct for biases from sampling

but std:
sample std: 0.35
bootstrap = 0.015 -> standard error

Estimated standard error -> std of bootstrap dist.
pop std.dev = std.error * sqrt(sample_size)

**Confidence Intervals**

using quantiles u can define ci

PDF integrates to CDF

plot Inv. CDF
`norm.ppf(quantile, loc = point_estimate_mean, scale = std_error)`

# Hypothesis Testing

## Hypothesis Testing Fundamentals

In this subchapter:
- Hypothesis tests and z-scores
- p-values
- Statistical significance

hypothesis: mean of pop is 110000
point esiimate: mean is 119000

generate bootstrap dist of sample means
- Resample, Calculate point estimate, save and repeat
`std_error = np.std(dist, ddof =1)`
standardized value = (value - mean) / std

z = (sample_stat - hypothesis) / standard error
within z standard error prob.

z-dist: mean 0 std: 1

**p-values:**
H tests check if sample statistic lies in tail of null dist.
p-value prob. of result given $H_0$ is true
-> large p-value for H0
3 types:
- left-tail 'fewer' `norm.cdf(z_score)`
- right-tail 'greater' `1 - norm.cdf(z_score)`
- two-tail 'different'

```python
p-value = 1-norm.cdf(z_score, loc = 0, scale = 1)
```
**Statistical Significance**
alpha: 0.05
if p <= 0.05 reject H0, else fail to reject H0

Condidence interval:
alpha -> 95% ci

Errors:

## Two-Sample and ANOVA Tests
In this subchapter:
- Performing t-tests
- Calculating p-values from t-statistics
- Paired t-tests
- ANOVA tests

**t-tests**
Compare between categories:
HO: mean same between categories
H1: mean is greater between categories

t = similiar to z.
$$t = \frac{(x_1 - x_2) - (\mu_1 - \mu_2)}{SE(x_1 - x_2)}$$

SE(x1 - x2) can bootstrap or error propagation
$SE(x1-x2) = \sqrt{\frac{s1^2}{n1} + \frac{s1^2}{n2}}$

can calculate t if assume H0 correct
has parameter df(dof)
large df -> normal dist.

z-statistic: used when using one sample statistic to estimate pop. parameter
t-statistic: used when using multiple statistics to estimate pop. parameter
```python
df.groupby('')[''].mean()
```
**p-value from t-value** 
`from scipy.stats import t`
3 types:
- left-tail 'fewer' `t.cdf(z_score)`
- right-tail 'greater' `1 - t.cdf(z_score)`
- two-tail 'different'

**paired t-tests**

dependent:
- want to capture patterns in datasets

```python
df['diff'] = df[column1] - df[column2]
```

$\longrightarrow$
$$t = \frac{(\mu_{diff})}{SE(x_{diff})}$$

$dof = n_{diff} - 1$

```python
p_value = t.cdf(t, df = n_diff -1)

```
t-value Easy ways
```python
import pingouin

pingouin.ttest(x = df['diff'].
               y = 0, # hypothesised values
               alternative = 'less','twosided','greater')

pingouin.ttest(x = df['x'].
               y = df['y'], # hypothesised values
               alternative = 'less','twosided','greater',
               paired = True/False)
```

performing unpaired t-tests on paired data increases chance of FN errors

**ANOVA tests**
tests for differences between groups

Visualise using box plots

alpha = 0.2

if set of 5 -> 10 pairewise tests
```python
pingouin.anova(data = df,
               dv = '', #dependent variable
               between = '' #column to compare between 
               )

pingouin.pairwise_tests(
  dv = '',
  between = '',
  padjust = 'bonf' # reduces chance of FP - corrects p-values
)
```
the more tests, quadratically increase pairs
the more tests, the higher chance one FP

## Proportion Tests

In this subchapter:
- One-sample proportion tests
- Two-sample proportion tests
- Chi-square tests of independence
- Chi-square goodness of fit tests



## Non-Parametric Tests
In this subchapter:
- Assumptions in hypothesis testing
- Non-parametric tests
- Non-parametric tests and unpaired t-tests
# Experimental Design in Python
## Experimental Design Preliminaries
In this subchapter:
- Setting up experiments
- Experimental data setup
- Normal data
## Experimental Design Techniques
In this subchapter:
- Factorial designs: principles and applications
- Randomised block design: controlling variance
- Covariate adjustment in experiential design
## Analyzing Experiment Data: Statistical Tests and Power
In this subchapter:
- Choosing the right statistical tests
- Post-hoc analysis following ANOVA
- P-values, alpha, and errors
- Power analysis: sample and effect size
## Advanced Insights from Experimental Complexity
In this subchapter:
- Synthesizing insights from complex experiments
- Addressing complexities in experimental data
- Applying nonparametric tests in experimental analysis



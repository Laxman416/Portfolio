# Machine Learning Techniques: Supervised and Unsupervised Learning <!-- omit in toc -->

**Course on Datacamp:**<br> 
`'Supervised Learning scikit-learn'`<br> 
`'Unsupervised Learning in Python'`<br>  
`'Machine Learning with Tree-Based Models in Python'`<br> 

- [Supervised Learning Scikit-learn](#supervised-learning-scikit-learn)
  - [Classification](#classification)
  - [Regression](#regression)
  - [Fine-Tuning Your Model](#fine-tuning-your-model)
  - [Preprocessing and Pipelines](#preprocessing-and-pipelines)

# Supervised Learning Scikit-learn

## Classification

In this subchapter:
- ML with scikit-learn
- Classification
- Measuring model performance

Unsupervised learning:
- Uncovering hidden patterns from unlabeled data - Clustering
Supervised learning:
- predicted values know, aim to predict values of unseen data
- give features (columns)

Two types of SL:
- Classification: Target variable consits of categories - could be binary
- Regression:  Target variable is continuous

Feature / predictor variables / independent variable
Target / response variable  / dependent variable

Requirements:
- No missing values
- Data in numeric format
- df or NumPy
- perform EDA

```python
from sklearn.module import Model

model = Model()
model.fit(X,y) # (features, target)
predictions = model.predict(X_new) #X_new is new data
```

**Classification SL**
- Build model and learn from labelled data
- pass unlabelled data and predit label

labelled data: training data

kNN(k-Nearest Neighbours)

Predict Label by:
    - looking at k closest labelled data points
    - Taking a majority vote
    - creates decision boundary
  
```python
from sklearn.neighbors import KNeighborsClassifier

X = df[featurs].values
y = df[target].values
# need same num of rows
knn = KNeighborsClassifer(n_neighbors = 15)
knn.fit(X,y)

predictions = knn.predict(2d_numpyarray)
```

**Measuring model performance**

Accuracy = correct / total

Split data into Training set and Test set
```python
from sklearn.model_selection import train_test_split

X_train, X_test,
y_train, y_test = train_test_split(X = ,
                                   y = ,
                                   test_size = 0.3, # size of test data
                                   random_state = int
                                   stratify = y) # Splits data to be representative of y

knn_score = knn.score(X_test,y_test)
```
Larger k: - underfitting
Smaller k: - overfitting (noise)

loop through multiple k values to find best one
```python
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1,26)
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors = neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)
```
Plot dictionary u can see peak test accuracy

## Regression

In this subchapter:
- Basics of linear regression
- Cross-validation
- Regularized Regression

predicting continuous values

Making predictions from single feature
```python
# Creating feature and target arrays
X = df.drop('target', axis = 1).values
y = diabetes['target'].values

# Making predictions from single feature
X_feature = X[:, int] # Slice column
X_feature = X_feature.reshape(-1,1) # need in 2D array

# Plot y against X
# Model
import sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_feature , y)
predictions = reg.predict(X_feature,y)

# plot model and data
```
Simple linear regression: uses one feature
Error Fn/loss Fn used

Line close to observation as possible
- calculate residual
- minimise residual sum of squares (OLS)
  
For higher dimensions:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test,
y_train, y_test = train_test_split(X = ,
                                   y = ,
                                   test_size = 0.3, # size of test data
                                   random_state = int
                                   stratify = y) # Splits data to be representative of y

reg_all = LinearRegression()
reg_all.fit(X_train,y_train)
y_pred = reg_all.predict(X_test)
# R^2: quantifies variance in target explained by features

reg_all.score(X_test,y_test) # Retirms R^2

## Using RMSE
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred, squared = False)
```

**Cross-validation**

R^2 dependent on way data is split.
use cross-validation
split data into 5 folds

![Split data into folds](image.png)

5 values of $R^2$

```python
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits = 6, shuffle = True, random_state = 42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=kf)
# calculate mean and std of cv
```

**Regularized Regression**

Linear Regression minimise Loss,
large coefficients -> overfitting

*Ridge regression:*
$$\text{Loss Fn} = \text{OLS Loss Fn} + \alpha \Sigma^n_i a_i^2$$

Ridge penalises large positive/negative values
$\alpha$: 
- hyperparameter to optimize model parameters
- =0 -> OLS -> overfitting
- high alpha underfitting


```python
from sklearn.linear_model import Ridge
scores = []
for alpha in [alpha_values_list]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    scores.append(X_test,y_test)

# Can see performance scores as alpha increases
```
*Lasso Regression:*
$$\text{Loss Fn} = \text{OLS Loss Fn} + \alpha \Sigma^n_i |a_i|$$

```python
# Same as Ridge
from sklearn.linear_model import Lasso
scores = []
for alpha in [alpha_values_list]:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    scores.append(X_test,y_test)
```

Lasso regression can be used to select important features of dataset
Shrinks coefficients of less important features to zero
Features not shrunk to zero selected by lasso

```python
# Lasso for feature selection
# Keep entire dataset

feature_names = df.drop('target', axis = 1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X,y).coef_

plt.bar(x= feature_names, y = lasso_coef)
```

## Fine-Tuning Your Model
In this subchapter:
- How good is model?
- Logistic Regression and ROC curve
- Hyperparameter tuning

Class imbalance:
causes high accuracy, but not good at predicting

Confusion Matrix for assessing classification performance

Accuracy: TP
Precision: TP/TP+FP (High precision = lower false positive rate)
Recall: TP/TP+FN (High recall = lower false negative rate)

F1 Score:
$$\text{F1 Score} = \frac{\text{precision} * \text{recall}}{\text{precision} + \text{recall}}$$

- equal weights to recall and precision
- favours models that have similiar recall and precision

```python
from sklearn.metrics import classification_report, confusion_matrix
knn = KNeighborsClassifier(n_neighbors = 7)
X_train, ....
knn.fit(X_train, y_train)

confusion = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
```

**Logistic regression**
- used for classification
- prob, p, observation belongs in binary classification
- if p > 0.5, data labeled as 1
- if p < 0.5, data labeled as 0

![Linear decision binary](image-1.png)

```python
from sklearn.liner_model import LogisticRegression

logreg = LogisticRegression()
X_train,...

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

y_pred_probs = logreg.predict_proba(X_test)[:, 1] # Selcet 2nd column out of 2D array
```
ROC curve:
varying thresholds:
![alt text](image-2.png)

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
# False positive rate, true positive rate
plt.plot(fpr,tpr)
```

ROC AUC: area under curve
ROC_AUC = 1 ideal
```python
from sklearn.mertrics import roc_aur_score

roc_auc_score(y_test, y_pred_probs)
```

***Hyperparameter Tuning*

- Ridge/Lasso Regression $\longrightarrow$   $\alpha$ 
- KNN                    $\longrightarrow$   n_neighbors 

Hyperparameter Tuning:
- Try multiple parameters, fit and compare
- use cross-validation to avoid overfitting

Grid search cross-validation
![alt text](image-3.png)
Choose the hyperparameter that perform best

```python
from sklearn.model_selection import GridSearchCV

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
params_grid = {"alpha": np.arange(0.0001,1,10),
                "solver": ["column1", "column2"]
}

ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_. ridge_cv.best_score_)
```

Limitations: on fold, hyperparameter, and values

Use RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
params_grid = {"alpha": np.arange(0.0001,1,10),
                "solver": ["column1", "column2"]
}

ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter = 2) #n_iter for 5 fold * n_iter  = 10 fits
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_. ridge_cv.best_score_)

test_score = ridge_cv.score(X_test, y_test)
```

## Preprocessing and Pipelines
In this subchapter:
- Preprocessing Data
- Handling missing Data
- Centering and scaling
- Evaluating multiple models

**Preprocessing Data:**
- Req. Numeric data and no missing data

scikit-learn will not accept categorical features.
need to convert into numeric
Convert to binary features by dummy variables

you can delete 1 column to avoid duplication of information

```python
df_dummies = pd.get_dummies(df[''], drop_first = True) 

df_dummies = pd.concat([music_df, music_dummies], axis = 1)
df_dummies = df_dummies.drop("categorical column",axis = 1)

# For one categorical variables
df_dummies = pd.get_dummies(df, drop_first = True)
linreg = LinearRegression()
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoring = "neg_mean_squared_error)
# returns -mse

rse = np.sqrt(-linreg_cv)
```

**Handling Missing Data**

Before Imputing must split data first otherwise data leakage

```python
df.isna().sum()

# if < 5% data drop data
df = df.dropna[subset = ['','']]

# Impute Missing Data
# mean/median or mods for categorical values

from sklearn.impute import SimpleImputer
X_cat = df['categorical variables'].values.reshape(-1,1)
X_num = music_df.drop(['cat','Target'], axis = 1).values
y = df[Target].values

# Create train and test set for cat and num
# Use random_state as same int for both. Target array values will be unchanged
X_train_cat, ...
X_train_num, ...

# Imputing Cat
imp_cat = SimpleImputer(stragey = 'most_frequent')
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

# imputing num
imp_num = SimpleImputer(stragey = '') # Default is mean
X_train_num = imp_cat.fit_transform(X_train_num)
X_test_num = imp_cat.transform(X_test_num)

X_train = np.append(X_train_cat, X_train_num, axis = 1)
X_test = np.append(X_test_cat, X_test_num, axis = 1)
```

Imputing with pipeline
```python
from sklearn.pipeline import Pipeline
#drop missing values thats < 5% data

df['Target'] = np.where(df[] == 'Target', 1,0) #convert to binary
X  = df.drop("Target", axis = 1).values
y = df[Target].values

## Building pipeline

steps = [("imputation", SimpleImputer()),
         ("logistic_regression", LogisticRegression())]

pipeline = Pipeline(steps)
X_train,...

pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```

**Centering and Scaling**

Models use some sort of dist.
Features on larger scale can disproportionally influence the model
e.g. KNN uses dist to make pred.

Scale data:
*Standardization*
- subtract mean divide by varaince
- features are centered around zero and have variance of 1

*Other scaling*
- subtract min divided by 1
- min 0 and maximum 1

- normalize data so between -1 and 1 


```python

df[['','']].describe()

from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis = 1).values
y = df['target'].values

X_train, ...

scal1r = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaled.fit_transform(X_test)

#or

steps = [('scaler', StandardScaler()).
         ('knn', KNeighborsClassifier())]

pipeline = Pipeline(steps)
X-train,...

knn_scaled = pipeline.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)

knn_scaled.score(X_test, y_test)

```

Cross Validation in pipeline
```python

steps = [('scaler', StandardScaler()).
         ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
X-train,...

parameters = {"knn_n_neighbors"}
X-train,...

cv = GridSearchCV(pipeline, param_grid = parameters)
cv.fit(X_train, y_train)

y_pred = cv.predict(X_test, y_test)
```

**Evaluating Multiple Models**

Which model?

Size of dataset:
- fewer features = simpler model, faster training time
- Some models like neural networks require large amount of data
  
Interpretibility:
- Some models easier to explain
- Linear regression

Flexibility:
- May improve accuracy by fewer assumptions about data

---
Regression Models performance:
- RMSE 
- R-squared

Classification model performance:
- Accuracy
- Confusion matrix
- Precision, recall, F1-score
- ROC AUC

Train several models and evaluate

---
Models affected by scaling:

- KNN
- LinearRegression (Ridge, Lasso)
- Logistic Regression
- Artificial Neural Networks

Do multiple models in for loop 
plot box plot
`for model in models.values()`

Also do models on test data
`for name, model in models.items()`
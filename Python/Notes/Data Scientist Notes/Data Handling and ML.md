# Data Handling and ML <!-- omit in toc -->

**Course on Datacamp:**<br> 
`'Intermediate Importing Data'`<br> 
`'Preprocessing in Machine Learning'`<br> 
`'Developing Python Packages'`<br> 
`'Machine Learning in Business'`<br> 

- [Intermediate Importing Data](#intermediate-importing-data)
  - [Importing Data from Internet](#importing-data-from-internet)
  - [Interacting with APIs to Import Data](#interacting-with-apis-to-import-data)
  - [Twitter API](#twitter-api)
- [Preprocessing in Machine Learning](#preprocessing-in-machine-learning)
  - [Introduction to Data Preprocessing](#introduction-to-data-preprocessing)
  - [Standardising Data](#standardising-data)
  - [Feature Engineering](#feature-engineering)
  - [Selecting Features for Modelling](#selecting-features-for-modelling)
- [Developing Python Packages](#developing-python-packages)
  - [From Loose Code to Local Package](#from-loose-code-to-local-package)
  - [Install your package](#install-your-package)
  - [Increasing package quality](#increasing-package-quality)
  - [Rapid Package Development](#rapid-package-development)
- [Machine Learning in Business](#machine-learning-in-business)
  - [Machine Learning and Data use cases](#machine-learning-and-data-use-cases)
  - [ML types](#ml-types)
  - [Business Requirements and Model Design](#business-requirements-and-model-design)
  - [Managing ML projects](#managing-ml-projects)

# Intermediate Importing Data

## Importing Data from Internet
In this subchapter:
- Importing flat files from web
- HTTP requests to import files
- Scraping the web in Python

**Importing flat files from web**

`urllib`
`urlopen()` -> accepts URLs instead of file names

```python
from urllib.request import urlretrieve

url = 'http://...'
urlretrieve(url, 'filename_to_write_to')
```

**HTTP requests to import files**

Protocol identifier: `http:`
Resource name: `''.com`
GET request -> go to website

```python
from urllib.request import urlopen, Request

url = ''
request = Request(url)
response = urlopen(request)

html = response.read()
response.close()

import requests

url = ''
r = requests.get(url)
text = r.text
```

**Scrapping Web**

HTML: mix of structured and unstructured
need to parse and extract structured data

```python
from bs4 import BeautifulSoup
import requests

url = ''
r = requests.get(url)
html_doc = r.text

soup = BeautifulSoup(html_doc) #prettifys html

title = soup.title
text = soup.get_text()

for link in soup.find_all('a'):
  print(link.get('href'))
```

## Interacting with APIs to Import Data
In this subchapter:
- Introduction to APIs and JSONs
- APIs and interacting with WWW

**Introduction to APIs and JSONs**

API: Application Programming Interface
Protocols and routines: Building and interacting with Software apps

JSON: file format
- JavaScript Object Notation
- Real-time server to browser
- Human readable
- Name value pairs like dictionary

```python
import json

with open('.json', 'r') as json_file:
  json_data = json.load(json_file)

for key, value in json_data.items():
  print(key + ':', value)
```

**API and interacting with WWW**

API: protocol and routines: allows two softwares to communicate

Connecting to API
```python
import requests
url = 'http://www.omdbapi.com/?t=hackers'
 # -> http request, www.omdbapi.com Querying OMDB API,
 # ?=hackers -> Query String, # Return data with title Hacker
r = requests.get(url)
json_data = r.json()
```

## Twitter API

In this subchapter:
- Twitter API and Authentication
  
Twitter API requires twitter account
Copy Access Tokens

REST API: allows read and write twitter data
Streaming API: read in realtime
GET statuses/sample: sample of streams
Tweets returned as JSON

```python
import tweepy, json

access_token = ''
access_token_secret = ''
consumer_key = ''
consumer_secret = ''

# Create Streaming Object
stream = tweepy.Stream(consumer_key, consumer_secret, access_token, access_token_secret)

# Filters Twitter Streams to capture data by keywords
stream.filter(track = ['word1', 'word2'])
```
# Preprocessing in Machine Learning

## Introduction to Data Preprocessing

In this subchapter:
- Working with data types
- Training and test sets

Data preprocessing:
- After EDA and Data cleaning
- preparing data for modelling
- data require numerical - dummy variables
- improve model performance

Techniques:
- Removing missing data
- `fillna`
- dummy variables
- `df.drop("column", axis = 1)` -> drop column if too much data missing
- `df.dropna(subset=['column_name'])`
- `df.dropna(thresh =2)` -> if 2 nan in same row remove

**Working with Data types:**
`.info()` -> gives data types
`df[''] = df[''].astype('float')`

**Training and Test Sets**

Why?:
- Reduces overfitting
- Evaluate performance
  
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = SEED, stratify = y) # proportional sampling -> Stratified
```

## Standardising Data

In this subchapter:
- Standardisation
- Log normalisation
- Scaling data for feature comparison
- Standardised data and modelling

**Standardisation:**

Why?
- transform continuous data to be normally dist.
- `scikit-learn` uses normally dist. data
- Log normalization and scaling

When to standardize:
- When Model uses linear space
- KNN, LinearRegression, KMeans Clustering
- When data has high-variance

- Features on different scales
- standardise features scales

**Log Normalisation**

When:
- When features have high variance
- Captures rel. change, mag. of change and everything positive

```python
import numpy as np

df['log_'] = np.log(df[''])
```

**Standardization data and modelling**

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#Avoid Data leakage by splitting first
X_train,...

knn = KNeighborsClassifier()
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.fit_transform(X_test)

knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)
```

## Feature Engineering

In this subchapter:
- Feature Engineering
- Encoding categorical variables
- Engineering numerical features
- Engineering text features

Extract and Expand info from existing information

**Encoding categorical variables**

```python
# binary

df[''] = df[''].apply(lambda val: 1 if val == "y" else 0)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
users[''] = le.fit_transform(df[''])

# Catergories

pd.get_dummies(df[''])
```
**Engineering numerical features**

For example Temp for different day, use mean instead.
Reduce Dimensionality.

Dates:
- convert to datetime
- extract component `dt.month`

**Engineering text features**

Extraction:
- numbers from string
- 
```python
import re

temp = re.search("\d+\.\d+", my_string) # extracts number
temp.group(0) # -> contains matches
```

TF/IDF: Vectorises words based upon importance
- TF: Term Frequency
- IDF: Inverse Document Frequency

```python
from sklearn.feature_extraction.text import TfidVectorizer

tfidf_vec = TfidVectorizer()

text_tfidf = tfidf_vec.fit_transform(document)
# Bayes Classifier peforms well on documents
# each feature independent
```

## Selecting Features for Modelling

In this subchapter:
- Feature selection
- Removing redundant features
- Selecting features using text vectors
- Dimensionality reduction 

**Feature Selection:**

Selecting features for modelling
Improve model's performance

reduce noise:
- e.g. too many geographical features
  
- remove duplicated features
- remove correlated features: features move together directionally
  but models assume feature independence
- remove noisy features

`df.corr()` -> if high only use 1

**Selecting features using text vectors**

- Select top 20% of variables
- Need to vary percentages

```python
tfidf_vec.vocabulary_ # vector of loc. description

text_tfidf[3].data # two components: word weight, index of word
text_tfidf[3].indicies

vocab = {v:k for k,v in tfidf_vec.vocabulary_.items()}

zipped_row = dict(zip(text_tfidf[3].indices, text_tfidf[3].data))

def return_weights(vocab, vector, vector_index):

  zipped = dict(zip(vector[vector_index],indices,
                    vector[vector_index].data))
  
  return {vocab[i]:zipped[i] for i in vector[vector_index].indices }
```

**Dimensionality reduction**

- Unsupervised learning method
- Combines/decomposes a feature space
- Feature extraction

PCA: Linear transformation to uncorrelated space
Captures as much variance as possible in each component

```python
from sklearn.decomposition import PCA
pca = PCA()

df_pca = pca.fit_transform(df) # for X, for X_test need to do .transform()
pca.explained_variance_ratio_ # -> Variance explained by each component
# Can drop other features
```

- Difficult to interpret components
- End of preprocessing journey

# Developing Python Packages

## From Loose Code to Local Package

In this subchapter:
- Starting a package
- Documentation
- Structuring imports

Allows to reuse code across files.
- Script - Python file
- Package - directory of Python files
- Subpackage - like `numpy.random`
- Module - A python file inside a package which stores package code
- Library - package or collection of packages

Directory:
- files
- `__init__.py` file: Initially empty, used to structure imports
- each subpackages require init files

**Documentation**

`help()` -> outputs documentation</br>
Doc string at top of Fn.

`pyment` -> generates docstring/ translates docstring templates

`pyment - w -o numpydoc file.py` -> Run on cmd line.

Documentation:
- Package: in init file  
- subpackage: in init file in subpackage directory
- module: at the top of module file

**Structuring Imports**

Without structure, can not import subpackage from the package

import Subpackage inside init file in package:
- Absolute import: `from mysklearn import preprocessing`
- Relative import: `from . import preprocessing`

for init file in preprocessing:
- `from mysklearn/preprocessing/normalize import \ normalize_data` imports the useful function

importing between sibling modules

`from .funcs import (func1, func2)`

## Install your package

In this subchapter:
- Installing packages
- Dealing with dependencies
- Including licenses and READMEs
- Publishing your package

`setup.py`

- Need an outer directory folder wil same name as package
- Outer directory contains setup.py

```python
from setuptools import setup

setup(
  author = '',
  description = '',
  name = '',
  version = '0.1.0',
  packages = find_packages(include = ['mysklearn', 'mysklearn.*'])
)
```

`pip install -e .` -> installs package

**Dealing with dependencies**

inside setup
```python
setup(
install_requires = ['pandas>1.0',
                    'scipy==1.1',
                    'matplotlib>2.2.1, <3']
python_requires = '>=2.7, !=3.0*, !=3.1*',
)
```
Making environments:

`pip freeze > requirements.txt`
`pip install -r requirements.txt`

**Publishing packages**

- downloading from PyPI, anyone can upload packages
- release early

Distributed package: bundled version of package
- Source Dist: mostly source code
- Wheel Dist: processed to make it faster
- upload both

`python setup.py sdist bdist_wheel`
- Creates `dist` directory.

Upload:
- `twine upload dist/*`
- `twine upload -r testpypi dist/*`


## Increasing package quality

In this subchapter:
- Testing your package
- Testing your package with different environments
- Keeping your package stylish

Testing along the way and save the code in a file.
Tracks down bugs

```python

def func(x):

  return x

def test_func():
  assert func(0) == 0
  # can do multiple asset lines
  
# Raises Assertion error if not true
```

- Test directory Copy structure of package
- create a test_module.py for each module.
- Create empty `__init__.py`
- test each function inside .py file

Run all at once using: 
`pytest`-> cmd line run at top of directory

**Testing with different environments**

`tox`

create `tox.ini` configuration file, top level of package

```python
[tox]

envlist = py27, py35, py36, py37

[testenv]
deps = pytest

commands = 
  pytest
  echo ...
```

**Formatting**

PEP8: variable and fn name guidance
`flake8` reads code

`flake8 file.py` cmd line

`#noqa` on line to avoid flake8 guidence
`#noqa: E222` avoids specific error on line

`flake8 --ignore E222 file.py`
`flake8 --select F401, F841 file.py`

Create a `setup.cfg` file to store settings

```cfg
[flake8]

ignore = E302
exclude = setup.py

per-file-ignores = 
  example/.../.py: E222
```

`flake8` run from terminal

## Rapid Package Development

In this subchapter:
- Faster package development with templates
- Version numbers and history
- Makefiles and classifiers

`cookiecutter`
- Creates empty Python packages
- All additional packages

`cookiecutter https://github.com/audreyr/cookiecutter-pypackage`

- Name:
- github_username:
- project_name:
- project_slug: #name used in pip install
- project_short_description:
- pypi_username: 
- use_pytest: y
- command_line_interface: 3

**Version numbers and history**

`CONTRIBUTING.md` md file to ask other developers to work on package</nr>
`HISTORY.md` What has changed between versions

```md
# History

## 0.3.0
### Changed
- changed feature

## 0.2.1
### Fixed
- Fixed bug

## 0.2.0
### Added
- feature added
```

Need to update version number for new release in:
  - top level `__init__.py`file : for the user .__version__
  - setup.py file: for pip
 
`bumpversion major/minor/patch` Run at top of directory in cmd line

**Makefiles and classifiers**

Classifiers:
- In setup.py
- allows users to find packages based on their environment

Makefiles:
- Commands for terminal
  
```makefile
dist: ## builds source and wheel package
  python setup.py sdist bdist_wheel

clean-build: ##remove build artifacts
  rm -fr build/
  rm -fr dist/
  rm -fr .eggs/

test: ## Run tests
  pytest

release: dist ## package and upload
  twine upload dist/*
```

`make fn_name` e.g. `make dist`
`make help`

# Machine Learning in Business

## Machine Learning and Data use cases

In this subchapter:
- ML and data pyramid
- ML principles
- Job roles, tools and technologies

**ML and data pyramid**

ML goals:
- Draw casual insights: Supervised ML
- Predict future events: Supervised ML
- Understand patterns in data: Unsupervised ML

![Data Hierarchy](../Images/ML%20for%20Business/Data%20Hierarchy.png)

**ML Principles**

Unsupervised ML: uses features and groups the rows into similar segments
- could be used to find anomalies

**Job roles, tools and tech.**

![Job Roles](../Images/ML%20for%20Business/Job%20Roles.png)

Team structures:
- Centralized: all data fn in one team
- Decentralized: each department has own data fn
- Hybrid: infrastructure, methods centralized while application and prototyping decentralized

## ML types

In this subchapter:
- Prediction vs inference dilemma
- Inference (casual) mode
- Predictions models (Supervised Learning)
- Prediction models (Unsupervised Learning)

**Prediction vs inference dilemma**

Inference/casual models:
- goal is to understand drivers of business outcome
- focused models are interpretable
- less accurate
- Which of the features affect target variable the most
  
Prediction:
- prediction is main goal
- more accurate but less interpretable

**Inference model**

Causality:
- identify causal relationship of how much certain actions affect an outcome
- answers 'why' questions
- experiments preferred over observational studies

**Supervised ML prediction models**

- predicting class
- predicting quantity

**Unsupervised ML prediction models**

- Clustering - grouping observations into similar groups
- Don't have Target variable
- Anomaly detection
- Recommender engines

## Business Requirements and Model Design

In this subchapter:
- Business requirements
- Model training
- Model performance measurement
- ML risks

**Business Requirements**

- Situation: Identify business situation
- Opportunity: Assess business opportunity: identify right markets
- Action: What actions to take: prioritise markets with higher predicted demand  

- Start with inference question.
- Build on inference question to define predictions questions

**Model Training**

Train on randomly sampled for model training
Test on new data data

Repeat process to find best model

**Model performance measurements**

Accuracy: --> classification
- Accuracy
- Precision
- Recall
  
Error: --> regression

if MSE too high can test non-linear models

**ML risks**

Poor performance on test:
- low precision: lots of FP
- low recall: missed lot of TP
- large error: for regression

Non-actionable model use cases:
- run tests to see if affects buisness otucomes
- A/B testing

![A/B Test](../Images/ML%20for%20Business/AB%20Test.png)

if tests don't work:
- increase data
- Build causal models to understand drivers
- Run qualitative research
- Change scope of problem: narrow or widen

## Managing ML projects

In this subchapter:
- ML mistakes
- Communication management
- ML in production

**ML mistakes**

- Shouldn't be ML first
- Not enough data
- Target variable definition
- Late testing, no impact
- Feature selection:
  - Inference: choose variables that can be controlled, and Business has to be involved in feature selection
  - Prediction: start with readily available data, then introduce new features iteratively.

**Communication Management**

Working groups:
- recurring meetings
- Define business requirements
  - What is business situation?
  - What is business opportunity?
  - What action to take?
- Review ML model
- inference vs prediction
- Baseline model results vs outline model updates
- Market testing
- Production
  - are tests consistent and model stable

Model performance and improvements:
- Classification: which class is more expensive to mis-classify
- Regression: error tolerance for prediction

**ML in production**

Production systems: live, customer facing and business critical

Prototype ML:
- Data Scientist
- ML Engineers

ML in production:
- Software Engineers
- Data Engineers
  

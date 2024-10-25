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

html = request.read()
response.close()

import requests

url = ''
r = request.get(url)
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

df_pca = pca.fit_transform(df)
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
## Install your package
In this subchapter:
- Installing packages
- Dealing with dependencies
- Including licenses and READMEs
- Publishing your package
## Increasing package quality
In this subchapter:
- Testing your package
- Testing your package with different environments
- Keeping your package stylish
## Rapid Package Development
In this subchapter:
- Faster package development with templates
- Version numbers and history
- Makefiles and classifiers

# Machine Learning in Business

## Machine Learning and Data use cases
In this subchapter:
- ML and data pyramid
- ML principles
- Job roles, tools and technologies
## ML types
In this subchapter:
- Prediction vs inference dilemma
- Inference (casual) mode
- Predictions models (Supervised Learning)
- Prediction models (Unsupervised Learning)
## Business Requirements and Model Design
In this subchapter:
- Business requirements
- Model training
- Model performance measurement
- ML risks
## Managing ML projects
In this subchapter:
- ML mistakes
- Communication management
- ML in production

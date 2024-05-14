# Introduction
Let me introduce the basics of building a machine learning model using Python. Python is widely used in the machine learning community due to its rich ecosystem of libraries and ease of use.
- Firstly, we select an IDE which typically includes a source code editor, build automation tools, and a debugger, all integrated into a single user interface used for writing code. In these projects will either use Pycharm or Jupiter Notebooks;
- Before getting started, ensure you have Python installed on your system along with the required libraries that will be using in our projects such as;
- Pandas: pandas DataFrame and performing any necessary preprocessing steps such as handling missing values and encoding categorical features
- Numpy: NumPy is a powerful Python library used for numerical computing. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.
- Matplotlib and Seaborn: Provides a wide variety of plots, including line plots, scatter plots, bar plots, histograms, and more.
- Scikit-learn: Scikit-learn is a machine learning library for Python that provides simple and efficient tools for data mining, data analysis and includes various machine learning algorithms for classification, regression, clustering, dimensionality reduction, and more.
## Table of Contents
- [Linear Regression](#linear-regression)
- [Logistic Regression](#logistic-regression)
- [Clustering](#clustering)

# Linear Regression
### Objective of the model
### Data Collection and Preparation
### Feature engineering
### Model Building
### Selection of algorithm
### Evaluation
- Model performance
- Insights from the results

# Logistic Regression
### Objective of the model
The objective of this model is to predict customer churn for a telecommunications company in order to identify at-risk customers and implement targeted retention strategies. 
### Data Collection and Preparation
Data was collected from the company's CRM system and included customer demographics, service usage, and billing information.
```python
#Data manipulation
import pandas as pd
import numpy as np

#Data Visualizations
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
```

# Machine learning libs
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import accuracy_score
```
### Feature engineering and processing
Preprocessing steps involved removing duplicates, imputing missing values, and one-hot encoding categorical features. Features such as total usage duration, average monthly spend, and tenure were engineered to provide additional predictive power based on domain expertise and exploratory data analysis.

```python
# Machine learning algorithms cannot work with categorical data directly, categorical data must be converted to number by
#the following method.

#Method: Dummy variable trap
#That is variables with only two values, zero and one all the categorical varibales at once with the code below


cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
data_scaled = scaler.fit_transform(std)

std_scaled = pd.DataFrame(data_scaled)

# Renaming columns of the stadadized colums
std_scaled.set_axis(['s_age', 's_duration', 's_pdays'], axis='columns', inplace=True)

#Concatenating the two dataframes
total_data = pd.concat([std_scaled , data_final], axis=1)

#Droping the columns 
```
### Model Building and Selection
Several classification algorithms were evaluated, including logistic regression, random forest, and gradient boosting. 
```python
# Splitting the data for trainng and testing with testing datasize of 20% and 80% for training

features = total_data.drop(['Y','s_age', 's_duration', 's_pdays'], axis=1)

X=features
y=data_final['Y']

# Splitting and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Model and fitting
logmodel = LogisticRegression()

#fitting the model
logmodel.fit(X_train, y_train)
```

### Prediction
The final model, a gradient boosting classifier, was selected based on its superior performance in terms of accuracy and interpretability. 
```python
# Predictions

pred=logmodel.predict(X_test)
pred

```
### Evaluation
- Model performance
The model achieved an accuracy of 85% on the test set, outperforming baseline models. However, further investigation is needed to address class imbalance and explore additional features that may enhance predictive performance.
```python
# Creating confutious matrix to check accuracy and wrong prediction

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, cmap='Greens')

cm=confusion_matrix(y_test, pred)
cm

![Matrix](https://github.com/Ackson507/Machine-Learning-ML-Projects/assets/84422970/b53bfcdc-7400-49a8-841e-7f30a571fc20)

print('Accuracy of the model is:',(4962/5519)*100)

Accuracy of 0.8990


```


### Model Application


# Clustering
### Objective of the model
### Data Collection and Preparation
### Feature engineering
### Model Building
### Selection of algorithm
### Evaluation
- Model performance
- Insights from the results





### References


# Introduction
Let me introduce the basics of building a machine learning model using Python. Python is widely used in the machine learning community due to its rich ecosystem of libraries and ease of use.
![Why-is-Python-the-Best-Suited-Programming-Language-for-Machine-Learning](https://github.com/Ackson507/Machine-Learning-ML-Projects/assets/84422970/2d38c43f-cc75-4267-89d8-3049c7ad0afb)


- Firstly, we select an IDE which typically includes a source code editor, build automation tools, and a debugger, all integrated into a single user interface used for writing code. In these projects will either use Pycharm or Jupiter Notebooks;
- Before getting started, ensure you have Python installed on your system along with the required libraries that will be using in our projects such as;
- Pandas: pandas DataFrame and performing any necessary preprocessing steps such as handling missing values and encoding categorical features
- Numpy: NumPy is a powerful Python library used for numerical computing. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.
- Matplotlib and Seaborn: Provides a wide variety of plots, including line plots, scatter plots, bar plots, histograms, and more.
- Scikit-learn: Scikit-learn is a machine learning library for Python that provides simple and efficient tools for data mining, data analysis and includes various machine learning algorithms for classification, regression, clustering, dimensionality reduction, and more.
## Table of Contents
- [Logistic Regression](#logistic-regression)
- [Linear Regression](#linear-regression)
- [Clustering](#clustering)


# Logistic Regression
### Objective of the model
The objective of this model is to predict customer churn for a telecommunications company in order to identify at-risk customers and implement targeted retention strategies. Machine learning model designed to predict whether a customer is likely to stop using a service or product, often referred to as "churn from customer at ZICK Bank.
### Data Collection and Preparation
Data was collected from the company's CRM system and included customer demographics, service usage, and billing information.
```python
#Data manipulation
import pandas as pd
import numpy as np

#Data Visualizations
Visualizing the number of people that subscribed

sns.countplot(x='Y', data=data, color='red')
plt.show()

data['Y'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'

no     88.8%
yes    11.2%
```

![Bar Chart](https://github.com/Ackson507/Machine-Learning-ML-Projects/assets/84422970/77962b83-90ea-4d91-83f1-ad67bf8a6721)



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
```
![Matrix](https://github.com/Ackson507/Machine-Learning-ML-Projects/assets/84422970/e6be3fe1-5bf0-4abf-bf69-9b193f4dd0a6)


```python
print('Accuracy of the model is:',(4962/5519)*100)

Accuracy of 0.8990

```
### Model Application
Machine learning model designed to predict whether a customer is likely to stop using a service or product, often referred to as "churn." Churn prediction models are valuable tools for businesses across various industries, including telecommunications, subscription-based services, e-commerce. Below are the uses;Customer Retention Strategies: Businesses can use churn prediction models to identify customers at high risk of churning and implement targeted retention strategies. This may include offering discounts, personalized promotions, or proactive customer support.

- Product Improvement: Insights from churn prediction models can help businesses understand the underlying reasons for churn and make improvements to their product or service. By addressing pain points and enhancing the customer experience, businesses can reduce churn rates.
- Resource Allocation: Churn prediction models enable businesses to allocate resources more efficiently by focusing efforts on customers with the highest likelihood of churn. This allows businesses to prioritize retention efforts and optimize marketing budgets.
- Subscription Management: In subscription-based businesses, churn prediction models help forecast future revenue and subscriber counts. By accurately predicting churn, businesses can adjust pricing strategies, subscription terms, and renewal incentives to maximize customer retention.
- Customer Segmentation: Churn prediction models can be used to segment customers based on their likelihood of churn and tailor marketing campaigns or loyalty programs to different customer segments. This personalized approach improves customer engagement and loyalty.

# Linear Regression
### Objective of the model
### Data Collection and Preparation
### Feature engineering
### Model Building
### Selection of algorithm
### Evaluation
- Model performance
- Insights from the results

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


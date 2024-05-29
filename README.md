# Introduction
Let me introduce this repository which demostrates the building blocks of machine learning model using Python language. Python is widely used in the machine learning community due to its rich ecosystem of libraries and ease of use.


- Firstly, I will select a suitable IDE which typically a source code editor good for building automation tools, and a debugger, all integrated into a single user interface used for writing code. In these projects will either use Visual Studio Code APP alongside with Jupiter Notebooks;
- Before getting started, ensure you have Python installed on your local machine. Then will configure it with our IDE will be working with.
- Pandas: pandas DataFrame and performing any necessary preprocessing steps such as handling missing values and encoding categorical features
- Numpy: NumPy is a powerful Python library used for numerical computing. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.
- Matplotlib and Seaborn: Provides a wide variety of plots, including line plots, scatter plots, bar plots, histograms, and more.
- Scikit-learn: Scikit-learn is a machine learning library for Python that provides simple and efficient tools for data mining, data analysis and includes various machine learning algorithms for classification, regression, clustering, dimensionality reduction, and more.
  
# Machine Learning Overview
A machine learning model is a representation of a system that has been trained to recognize patterns or make decisions based on input data. It is the end product after a machine learning algorithm has been trained on a dataset. On the other hand what is inside the model is an algorithm, a method or a set of rules followed to solve a problem, especially by a computer.In the context of machine learning, it is a procedure or formula for training a model on a dataset. 
There are various types of machine learning algorithms, such as: 
- Supervised Learning Algorithms: These require labeled data and include algorithms like linear regression, logistic regression, support vector machines, and neural networks.
- Unsupervised Learning Algorithms: These work with unlabeled data and include algorithms like k-means clustering, hierarchical clustering, and principal component analysis (PCA).
- Reinforcement Learning Algorithms: These involve training models to make sequences of decisions by rewarding them for good decisions and penalizing them for bad ones, such as Q-learning and deep reinforcement learning.

### Problem Statement
Banks often conduct marketing campaigns to promote their products and services, such as term deposits, to their customers. Understanding which customers are likely to subscribe to a term deposit can significantly enhance the effectiveness of these campaigns, optimize marketing efforts, and increase the overall return on investment. The challenge is to predict whether a customer will subscribe to a term deposit based on various input features collected during past marketing campaigns.

### Objective of the model
The objective of this project is to develop a machine learning model that can predict whether a bank customer will subscribe to a term deposit based on the data from previous marketing campaigns.

### Data Collection: 
We collect historical data from previous marketing campaigns, including customer demographics, contact history, and campaign outcomes. First and foremost we install the libraries needed using pip, if already installed we proceed to import them into our workig enviroment.

```python
#Data Wrangling
import pandas as pd
import numpy as np

#Data visualizations
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')
```
We then load the dataset;

```python
df = pd.read_csv("Campaing.csv")
```
![Screenshot (625)](https://github.com/Ackson507/Machine-Learning-Algorithms/assets/84422970/83315392-c7df-4db6-b620-bf2ef44be8cc)


### Data Preprocessing and Feature Engineering
We the clean and preprocess the data to handle missing values, outliers, and inconsistencies. This is where we spend most time because if the dataset we are training on is not properly prepared if will affect the later stage of model and its performance. Feature engineering is a crucial step in the machine learning pipeline where raw data is transformed into meaningful features that can enhance the performance of predictive models. The goal is to create a set of input variables that provide the most predictive power for the model.
- Creating New Features
- Transforming Existing Features such as Encoding of convert categorical variables like job, marital, and education into encoded columns.
- Converting categorical columns into dummy variables, also known as one-hot encoding, is a common preprocessing step in machine learning. This process transforms categorical data into a format that can be provided to machine learning algorithms to improve their performance

```python
data['Y'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'

no     88.8%
yes    11.2%
```

![Bar Chart](https://github.com/Ackson507/Machine-Learning-ML-Projects/assets/84422970/77962b83-90ea-4d91-83f1-ad67bf8a6721)

```python
#1 Check columns list and missing values and dropping null rows
df.isnull().sum()

#2 Let us group “basic.4y”, “basic.9y” and “basic.6y” together and call them “basic”.

df['education']=np.where(df['education'] =='basic.9y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.6y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.4y', 'Basic', df['education'])
```
```python
# Applying one-hot encoding to categorical varibales.

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df[var], prefix=var)
    data1=df.join(cat_list)
    df=data1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


```

### Dataset Splitting and model Selection: 
Choose appropriate machine learning algorithms for binary classification. Common algorithms include logistic regression, decision trees, random forests, gradient boosting machines, and support vector machines.
```python
# We begin by Importing the machine learning libraries now

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Model 1
from sklearn.svm import SVC                         # Model 2
from sklearn.linear_model import LogisticRegression # Model 3
from sklearn.tree import DecisionTreeClassifier     # Model 4
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score # Model Evaluation

#Creating Test and Train data
df_final['is_Train']=np.random.uniform(0, 1, len(df)) <= .80

#Creating data for train and test rows
train, test = df_final[df_final["is_Train"]==True], df_final[df_final["is_Train"]==False]

#Creating data for train and test rows
train, test = df_final[df_final["is_Train"]==True], df_final[df_final["is_Train"]==False]

#Removing two columns not acting as features
df1=df1.drop(['Y'])

#Creating a list of feature names INDEPENDENT variables
features = df1

#Target outcome DEPENDANT varibale
y=train['Y']
```

### Model fitting and Training : 
Train the model on the training data. In this below code will create  a variable called models, this valuable will hold the algorithm we have selected to train the dataset on and at the end of the project will evaluate and see which model performs better depending on the evaluation metrics. Below is the list of models I have selected;
- Random Forest
- Support Vector Machine
- Logistic Regression
- Decision Trees

Then after selecting these algorithms, we fit the dataset to the models and use " for Loop" functional to iterate the process for all models and give metrics for each one.
```python

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Assuming df_final is your complete DataFrame and 'is_Train' is the column used for splitting
train, test = df_final[df_final["is_Train"]==True], df_final[df_final["is_Train"]==False]

# Ensure df1 is a list of feature names
# Example: features = ['feature1', 'feature2', 'feature3', ...] 
features = df1  # Ensure df1 is defined correctly as a list of feature names

# Convert target labels to numeric values
train['Y'] = train['Y'].map({'no': 0, 'yes': 1})
test['Y'] = test['Y'].map({'no': 0, 'yes': 1})

# Define the target variable
target = 'Y'
y_train = train[target]
y_test = test[target]

# Initialize the models
models = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier()
}

# Evaluate each model
results = {}
for model_name, model in models.items():
    model.fit(train[features], y_train)
    y_pred = model.predict(test[features])
    y_proba = model.predict_proba(test[features])[:, 1] if hasattr(model, "predict_proba") else None
    
    results[model_name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"
    }

# Print the results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")
    print()

```

### Model Evaluation: 
Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1-score in predicting customer subscriptions. 
- Best model was Logistic Regression with 90.08% accuracy
- Followed by Support Vecotor Machine with 89.89%,
- Random Forest with 88.72% accuracy
- Decission Trees with 85.63% accuracy
  
![Screenshot (626)](https://github.com/Ackson507/Machine-Learning-Algorithms/assets/84422970/2d040ae5-2906-44c5-8182-66ccdb113b29)



### Deployment and Monitoring: 
When we have multiple model selected for a project, it is obvious we will move forward with best performing model. After selecting the model then we start modelling it to improve the performance such as;
- Hyperparameter Tuning: Grid Search, Random Search and Bayesian Optimization ers, balancing exploration and exploitation to find optimal values more efficiently than grid or random search.
- Feature Engineering:Creating New Features, Feature Transformation and Feature Selection.
- Ensembling Methods: Bagging, Boosting and Stacking.

- Experiment with Advanced Models: Try more complex models like neural networks, especially if you have a large dataset, or ensemble methods like XGBoost and LightGBM.
Transfer Learning: Use pre-trained models and fine-tune them on your specific dataset, especially in domains like image or text classification.
- Early Stopping: Monitor the model’s performance on a validation set and stop training when performance stops improving.
-  Increasing Training Data: Gather more data if possible, as more training data can improve model performance.

Then later deploy the model into the bank's operational environment. Continuously monitor its performance and retrain it periodically with new data to maintain accuracy.

### Application
- Targeted Marketing Campaigns: By predicting which customers are likely to subscribe, the bank can focus its marketing efforts on those customers, improving the efficiency and success rates of the campaigns.
- Personalized Offers: The bank can use the model's predictions to tailor offers and communications to individual customers based on their likelihood of subscribing, increasing the relevance and attractiveness
   of the offers.
- Resource Allocation: Allocate marketing resources more effectively by concentrating on high-potential customers, thereby optimizing marketing budgets and efforts.
- Customer Insights: Gain insights into customer behavior and preferences, helping the bank to refine its overall marketing strategy and product offerings.


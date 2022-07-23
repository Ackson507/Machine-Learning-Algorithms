#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PROJECT NO: 1

#Project Name: ANNUAL STORE SALES ANALYSIS 2015, 2016 and 2017.
#Tool: Python and Power BI
#AIM: With the help of visualization, we will answer the business objectives for analysis of the company sales.

#BUSINESS OBJECTIVES
#1. Which year had the most sales or revenue for the company? = 2018
#2. What category was most at demand or sold during the past years? =  Office Supplies
#3. What segment was the highest for the sold products = Consumer
#4. Which ship mode was used mostly = Standard Class
#5.What was the rate of change of revenue from 2015 - 2016 =  -3.5%
#                                              2016 - 2017 =  29.8%
#                                              2017 - 2018 =  20.5% 
#6 What was the total sales made in the period of 2015 to 2019 =

#7. By help of our BI tool such as Power BI, we will create a dashboard to further visualization to help us answer our business 
# objectives and assist with more insights, pattern and trends of the sales in each particular year.


# In[2]:


import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np


# In[3]:


data = "C:/Users/Ackson/Desktop/DATASETS/Store_Sales_2015-2017.csv"
storedata = pd.read_csv(data)
#print(storedata)


# In[4]:


# Our new dataset will store it in new 
# At this point we co
new = storedata.drop(["ShipDate","Order ID", "Customer ID", "Country", "Postal Code", "Product ID", "Product Name","Customer Name","Region","State"], axis = 'columns')
new


# In[5]:


#Before working with dates, I need to format the date column into date

new['OrderDate']= pd.to_datetime(new['OrderDate'])


# In[6]:


sales15_18 = new['Sales']
sales15_18.sum


# In[7]:


#Now we filter and lock the dataset into years as labeled below

new_2018 = new.loc[(new['OrderDate'] >= '2018-01-01')
                     & (new['OrderDate'] < '2018-12-31')]
new_2015 = new.loc[(new['OrderDate'] >= '2015-01-01')
                     & (new['OrderDate'] < '2015-12-31')]
new_2017 = new.loc[(new['OrderDate'] >= '2017-01-01')
                     & (new['OrderDate'] < '2017-12-31')]
new_2016 = new.loc[(new['OrderDate'] >= '2016-01-01')
                     & (new['OrderDate'] < '2016-12-31')]


# In[8]:


# We find the total sales that was in 2015, 2016, 2017, and 2018
Total_2015 = new_2015['Sales'].sum()
Total_2016 = new_2016['Sales'].sum()
Total_2017 = new_2017['Sales'].sum()
Total_2018 = new_2018['Sales'].sum()


# In[9]:


# Q1 From the above code we can see that 2018 hard the most sales.
print("Sales 2015 is $474,602")
print("Sales 2016 is $458,054")
print("Sales 2017 is $599,460")
print("Sales 2018 is $722,052")


# In[10]:


# We have cleaned the dataset and remain with the required column to start the anlysis to answer the business objectives
#Q2 ANSWER
#Most sold category

# Using groupby() and count()
new2 = new.groupby(['Category'])['Category'].count()
new2
# From the below output we can conclude that most sold was office funitures


# In[11]:


# Ship mode most used by customers
new3 = new.groupby(['Ship Mode'])['Ship Mode'].count()
new3


# In[12]:


# Q4 We find the frequest most sold Segment is Consumer
new4 = new.groupby(['Segment'])['Segment'].count()
new4


# In[13]:


#Q6
Total = ((Total_2018-Total_2017)/Total_2017)*100
print("Rate of change from 2017 to 2018 is 29.8% ")


# In[14]:


Total


# In[16]:


new.to_csv("cleaned Storedataset_.csv")


# In[ ]:





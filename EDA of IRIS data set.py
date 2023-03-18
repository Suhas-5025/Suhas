#!/usr/bin/env python
# coding: utf-8

# Dataset Information
# The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
# 
# Attribute Information:
# 
# sepal length in cm
# sepal width in cm
# petal length in cm
# petal width in cm
# class:
# -- Iris Setosa -- Iris Versicolour -- Iris Virginica

# Import modules

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Loading the dataset

# In[2]:


df = pd.read_csv('Iris.csv')
df.head()


# In[3]:


# delete a column
df = df.drop(columns = ['Id'])
df.head()


# In[4]:


# to display stats about data
df.describe()


# In[5]:


# to basic info about datatype
df.info()


# In[6]:


# to display no. of samples on each class
df['Species'].value_counts()


# Preprocessing the dataset

# In[7]:


# check for null values
df.isnull().sum()


# Exploratory Data Analysis

# In[8]:


# histograms
df['SepalLengthCm'].hist()


# In[9]:


df['SepalWidthCm'].hist()


# In[10]:


df['PetalLengthCm'].hist()


# In[11]:


df['PetalWidthCm'].hist()


# In[12]:


# scatterplot
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[13]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[14]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[15]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[16]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# Coorelation Matrix
# A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. The value is in the range of -1 to 1. If two varibles have high correlation, we can neglect one variable from those two

# In[17]:


df.corr()


# In[18]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# Label Encoder
# In machine learning, we usually deal with datasets which contains multiple labels in one or more than one columns. These labels can be in the form of words or numbers. Label Encoding refers to converting the labels into numeric form so as to convert it into the machine-readable form

# In[19]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[20]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# Model Training

# In[22]:


from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[23]:


# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[24]:


# model training
model.fit(x_train, y_train)


# In[25]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[26]:


# knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[27]:


model.fit(x_train, y_train)


# In[29]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[30]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[31]:


model.fit(x_train, y_train)


# In[32]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[ ]:





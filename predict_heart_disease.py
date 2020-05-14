# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:59:56 2020

@author: LENOVO
"""
#import necessary libraries
import pandas as pd
import numpy as np

#reading data set 
data=pd.read_csv('D:\ML Project\heart.csv')

data.info()

#Exploratory Data Analysis (EDA)
print(data.shape)  #data shape i.e. rows and column

print(data.target.value_counts())  #counting the target value

# Checking for messing values
print(data.isna().sum())  # checking null value

#slicing data into dependent and independent variaata les

X = data.iloc[:,0:13].values
y = data.iloc[:,-1].values

#Spiliting data into train and test
from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Applying Logistic Regression algorithm and fitting train data in the model
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(X_train, y_train)

#predicting the unseen data 
y_pred=lg.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

#accuracy score
from sklearn.metrics import accuracy_score
acc= accuracy_score(y_test,y_pred)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 08:50:26 2019

@author: ajodas
"""

import pandas as pd
import numpy as np

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

dataset = pd.read_csv ('BankNote.csv' , header = None)
dataset.columns = ['variance' , 'skewness' , 'kurtosis' , 'entropy' , 'class']
print (dataset.head())
print (dataset.shape)

X = dataset.drop('class', axis=1)  
y = dataset['class']

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 

 
classifier = DecisionTreeClassifier ()
print (classifier)
classifier.fit(X_train, y_train)  


y_pred = classifier.predict(X_test)  

 
print(confusion_matrix(y_test, y_pred)) 
print ("************") 
print(classification_report(y_test, y_pred))  

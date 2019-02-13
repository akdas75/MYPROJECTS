#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:54:52 2019

@author: ajodas
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression 

import matplotlib.pyplot as plt

print ("Linear Regression using Matrices ");

IN = np.array ([4.0,4.5,5.0,5.5,6.0,6.5,7.0])
#X = np.array ([4,5,6,7,8,9,10])
OUT = np.array ([33,42,45,51,53,61,62])

plt.scatter (IN,OUT)
plt.ylabel ("Dependent Variable")
plt.xlabel ("Independent Variable")
plt.show()

X = IN[:,np.newaxis]
Y = OUT[:,np.newaxis]

print ("Data size {}". format (X.size))
print ("Data shape {}". format (X.shape))

data_size = X.size

iterations = 8000
initial_b  = 0 
initial_m  = 0
learning_rate = 0.01

print (data_size)

def compute_gradient(m ,b):
    
    gradient_b = 0
    gradient_m = 0
    for i in range (data_size):
        x = IN[i]
        y = OUT[i]
        gradient_b += (-2/data_size) * (y - (m * x + b))
        gradient_m += (-2/data_size) * x * (y- (m * x + b))
        
    new_m = m - (gradient_m * learning_rate)
    new_b = b - (gradient_b * learning_rate)
    
    return (new_m,new_b)

m = initial_m
b = initial_b

for i in range (iterations):
    print ("iteration {} m {} b {}".format(i, m, b))
    [m,b] = compute_gradient(m ,b) 
    
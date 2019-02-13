#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:23:08 2019

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

print ("output size {}". format (Y.size))
print ("output shape {}". format (Y.shape))

mean_x = np.mean(X)
mean_y = np.mean(Y)

XT = X.transpose()
YT = Y.transpose()

COV_XY = np.cov (XT, YT)[0][1]
VAR_X= np.var (XT, ddof=1)
print ("COV_XY {} ". format (COV_XY))
print ("VAR_X {} ". format (VAR_X))


#calculate the coefficients
b0 = COV_XY/VAR_X
b1 = mean_y - (b0 * mean_x)  


print("\nCoefficients {}, {}".format (b1, b0))

reg_line = [(b0*x) + b1 for x in IN]

print (reg_line)
reg_line_a = np.asarray(reg_line)
reg_line_a= reg_line_a[:,np.newaxis]

plt.scatter (IN, OUT, color='red')
plt.plot (X, reg_line)
plt.ylabel ("Dependent Variable")
plt.xlabel ("Independent Variable")
plt.title  ("Regression line")
plt.show()

print (reg_line_a.shape)
print (Y.shape)
error = np.subtract (reg_line_a , Y)
print (error)


"""
Lets verify via sklearn
"""

print (" From sklearn")

regressor = LinearRegression()  
regressor.fit(X, Y)  
print(regressor.intercept_) 
print(regressor.coef_)  
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:58:34 2019

@author: ajodas
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression 

import matplotlib.pyplot as plt

import scipy.stats as sp_stats

# https://www.youtube.com/watch?v=dQNpSa-bq4M&t=135s

print ("Multiple Linear Regression using Matrices ");

"""
IN1 is the miles travelled
IN2 is the number of deliveries
IN3 is the gas price in dollars
y is the travel time in hours


Multiple regression is many to one relationship


Two or more independent variables are used to predict/explain the variance
in one dependent variable

Adding more independent variables to the multiple regression procedure doesnot
mean the regression will be better or offer better predictions. In fact
it can make things worse. This is called overfitting

The addition of more independent variables create more relationship among them
So not only the indenpendent variables poterntially related to the dependent
variables they are also potentially related to each other. When this happens
it is called multi collinearity 

The ideal is all the independent variables to be corelated with the 
dependent variables but not with each other

y =b0 + b1x1 + b2x2 + ..... + error

"""
def plot_scatter (X , Y, x_label, y_label):
    plt.ylabel (y_label)
    plt.xlabel (x_label)
    plt.scatter (X,Y)
    plt.show()

def compute_cor_p_value (X1, X2, df) :
    corelation= np.corrcoef(X1,X2)
    t = (corelation[0][1] * np.sqrt(df)) / (np.sqrt(1 - (np.square(corelation[0][1]))))
    p = (1 - sp_stats.t.cdf(abs(t), df)) * 2
    print (" t {}".format (t))
    print (" p %f" %p)
    return (corelation, p)

def compute_coeffs (X , Y):
    #print ("X shape {}".format(X.shape))
    r,c = X.shape
    K = np.ones (shape = (r,1))
    #print ("K shape {}".format(K.shape))
    A = np.concatenate ((K , X), axis = 1)
    #print ("Matrix A \n {} \n".format(A))
    A_Transpose = A.transpose()
    #print (" A<transpose> \n {} \n".format (A_Transpose))
    M = A_Transpose.dot(A)
    #print  ("A<transpose>.A \n {} \n". format (M))
    M_inv = np.linalg.inv (M)
    #print ("inv (A <transpose>.A) \n {} \n". format (M_inv))
    N = A_Transpose.dot (Y)
    #print ("A<transpose>.b \n {} \n".format (N))
    coeffs = M_inv.dot (N)
    #print ("Coefficients \n {} \n". format (coeffs))
    #print (coeffs.shape)
    #intercept,slope = coeffs[0][0], coeffs[1:,:]
    #intercept,slope = coeffs
    intercept , slope = coeffs[0][0] , coeffs[1: , 0:]
    return (slope , intercept)

def plot_multiple_regression_line (X, Y, slope, intercept, x_label, y_label):  
    return
  
    
def plot_regression_line (X, Y, slope, intercept, x_label, y_label):  
    reg_line = [(slope*x) + intercept for x in X]
    plt.scatter (X, Y, color='red')
    plt.plot (X, reg_line)
    plt.ylabel (y_label)
    plt.xlabel (x_label)
    plt.title  ("Regression line")
    plt.show()
  
def compute_sklearn_coeff (X, Y):
    print (" From sklearn")
    regressor = LinearRegression()  
    regressor.fit(X, Y)  
    print(regressor.intercept_) 
    print(regressor.coef_)  
    
def compute_single_regression (X, Y, x_label, y_label, confidence_level):
    slope,intercept = compute_coeffs (X, Y)
    print ("Slope  :  {} ".format(slope))
    print ("Intercept : {} \n".format(intercept))

    plot_regression_line (X, Y, slope[0], intercept, x_label, y_label)
    compute_sklearn_coeff (X, Y)
    
    
def compute_multiple_regression (X, Y, confidence_level):
    slope,intercept = compute_coeffs (X, Y)
    print ("Slope  :  {} ".format(slope))
    print ("Intercept : {} \n".format(intercept))
    
    x_label = "Independent Variable"
    y_label = "Dependent Variable"
    plot_multiple_regression_line(X, Y, slope, intercept, x_label, y_label)   
  
    
# *********************************************************************** #

IN1 = np.array ([89,66,78,111,44,77,80,66,109,76])
IN2 = np.array ([4,1,3,6,1,3,3,2,5,3])
IN3 = np.array ([3.84,3.19,3.78,3.89,3.57,3.57,3.03,3.51,3.54,3.25])
OUT = np.array ([7,5.4,6.6,7.4,4.8,6.4,7,5.6,7.3,6.4])

print ("Data 1 size {}". format (IN1.size))
print ("Data 2 size {}". format (IN2.size))
print ("Data 3 size {}". format (IN3.size))
print ("Outout size {}". format (OUT.size))


plot_scatter (IN1, OUT, "Miles Travelled", "Travel Time")
plot_scatter (IN2, OUT, "Number of Deliveries", "Travel Time")
plot_scatter (IN3, OUT, "Gas Price", "Travel Time")

"""
Gas price donot have linear relationship with the travel time (output variable) 
Other two independent variable has linear relationship with the 
dependent/output variable
so we dont need gas price for linear regression model
"""
"""
Check for multicolinearity
Miles travelled  and travel time have linear pattern with each other and
cause serious problems. We neednot use both as they are redundant
"""


plot_scatter (IN1, IN2, "Miles Travelled", "Number of Deliveries")
plot_scatter (IN2, IN3, "Number of Deliveries", "Gas Price")
plot_scatter (IN3, IN1, "Gas Price", "Miles Travelled")

df = IN1.size


corelation , p= compute_cor_p_value (IN1,OUT,df)
print ("Pearson corelation coeff miles travlled vs travel time {}".format(corelation[0][1]))

corelation, p= compute_cor_p_value (IN2,OUT,df)
print ("Pearson corelation coeff num deliveries vs travel time {}".format(corelation[0][1]))

corelation , p= compute_cor_p_value (IN3,OUT,df)
print ("Pearson corelation coeff gas price vs travel time {}".format(corelation[0][1]))

corelation , p= compute_cor_p_value (IN1,IN2,df)
print ("Pearson corelation coeff miles travlled vs num deliveries {}".format(corelation[0][1]))

corelation , p= compute_cor_p_value (IN2,IN3,df)
print ("Pearson corelation coeff num deliveries vs gas price {}".format(corelation[0][1]))

corelation , p= compute_cor_p_value (IN1,IN3,df)
print ("Pearson corelation coeff miles travlled vs gas price {}\n".format(corelation[0][1]))


X1 = IN1[:,np.newaxis]
X2 = IN2[:,np.newaxis]
X3 = IN3[:,np.newaxis]
Y =  OUT[:,np.newaxis]

print ("*****************************")
print ("Single Linear Regression : IV:Miles Travelled DV:Travel time")
compute_single_regression (X1,Y , "Miles Travelled", "Travel time" ,95)


print ("*****************************")
print ("Single Linear Regression : IV:Num Deliveries DV:Travel time")
compute_single_regression (X2,Y , "Num Deliveries","Travel time", 95)

print ("*****************************")
print ("Single Linear Regression : IV:Gas Price DV:Travel time")
compute_single_regression (X3,Y , "Gas Price","Travel time",95)
print ("*****************************")

print ("Multiple Linear Regression : IV:Miles Travelled,Num Deliveries DV:Travel time")
X12 = np.hstack ([X1 ,X2])
compute_multiple_regression (X12 , Y, 95)
print ("*****************************")


print ("Multiple Linear Regression : IV:Miles Travelled,Gas Price DV:Travel time")
X13 = np.hstack ([X1 ,X3])
compute_multiple_regression (X13 , Y, 95)
print ("*****************************")

print ("Multiple Linear Regression : IV:Num Deliveries,Gas Price DV:Travel time")
X23 = np.hstack ([X2 ,X3])
compute_multiple_regression (X23 , Y, 95)
print ("*****************************")

print ("Multiple Linear Regression : IV:Miles Travelled, Num Deliveries ,Gas Price DV:Travel time")
X123 = np.hstack ([X1 ,X2, X3])
compute_multiple_regression (X123 , Y, 95)
print ("*****************************")

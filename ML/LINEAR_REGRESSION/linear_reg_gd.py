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

def cost (m,b , data_size):
    x = IN
    y = OUT
    totalError = 0
    for i in range (data_size):
        x = IN[i]
        y = OUT[i]
        totalError += ((m*x + b) - y) ** 2
    return totalError/ float(data_size)
     

def compute_gradient(X , Y, theta_1 ,theta_0 , N, learning_rate):
    
    gradient_theta_0 = 0
    gradient_theta_1 = 0
    
    #print (X.shape, Y.shape, N)
          
    Y_pred = theta_1*X + theta_0
       
    gradient_theta_1 = ((-2/N) * sum(X * (Y - Y_pred)))
    gradient_theta_0 = ((-2/N) * sum(Y - Y_pred))
        
      
    #print (gradient_theta_0 , gradient_theta_1, gradient_theta_0 * learning_rate, gradient_theta_1 * learning_rate)    
    new_theta_0 = theta_0 - (gradient_theta_0 * learning_rate)
    new_theta_1 = theta_1 - (gradient_theta_1 * learning_rate)
    
    return (new_theta_1,new_theta_0)

IN = np.array([63 , 64, 66, 69, 69, 71, 71, 72, 73, 75])
OUT = np.array([127,121,142,157,162,156,169,165,181,208])

print (IN.shape)
print (OUT.shape)

fig = plt.figure()
ax2 = fig.add_subplot(121)
ax3 = fig.add_subplot(122)

ax2.scatter (IN,OUT)
ax3.scatter (IN,OUT)

X = IN[:,np.newaxis]
Y = OUT[:,np.newaxis]

plt.plot (X,Y)


print ("Data size {}". format (X.size))
print ("Data shape {}". format (X.shape))

data_size = X.size

iterations       = 1000
initial_theta_0  = 0 
initial_theta_1  = 0
learning_rate    = 0.00001  
theta_0          = initial_theta_0
theta_1          = initial_theta_1

fig,ax = plt.subplots(figsize=(12,8))
cost_history = []

for i in range (iterations):
    #print ("iteration {} m {} b {}".format(i, theta_1, theta_0))
    [theta_1, theta_0] = compute_gradient(X , Y , theta_1 ,theta_0, data_size, learning_rate) 
    totalError = cost (theta_1,theta_0, data_size)
    #print (totalError)
    cost_history.append (totalError)
    
ax.plot(range(iterations),cost_history,'b.')    

print ("iteration {} m {} b {}".format(i, theta_1, theta_0))

reg_line = [(theta_1 * x) + theta_0 for x in IN]

lm = LinearRegression()
lm.fit(X, Y)

print ("SKLEARN coeff {}".format(lm.coef_))
print ("SKLEARN intercept {}".format(lm.intercept_))


#reg_line = [(lm.coef_[0] * x) + lm.intercept_ for x in IN]

ax3.plot (IN, reg_line , color='red')  
plt.show()



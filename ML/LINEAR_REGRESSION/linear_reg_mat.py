#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 08:22:40 2019

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

"""
Linear Algebra is great but it can not solve 6 equations
with 2 number of unknows
Atleast most of the cases.

"""

"""
So now the equations are 

    C + 4.0 D = 33
    C + 4.5 D = 42
    C + 5.0 D = 45
    C + 5.5 D = 51
    C + 6.0 D = 53
    C + 6.5 D = 61
    C + 7.0 D = 62
    
The ask is to find the best fitting line
        
       
        |  1   4.0  |  |     |       |  33  | 
        |  1   4.5  |  |     |       |  42  |
        |  1   5.0  |  |  C  |       |  45  |
    A = |  1   5.5  |  |     |    =  |  51  |
        |  1   6.0  |  |  D  |       |  53  |
        |  1   6.5  |  |     |       |  61  |
        |  1   7.0  |  |     |       |  62  |

   
   The equation Ay = b is only solvable when b is in the
   column space of A.
   
   if b is not in the column spce of A we dont have a solution
   
   So the other way is to project b onto the column space of A
   so that p = y(hat)A
   
   p is the projection onto the colum space of A.
   y (hat) is the estimate [C,D]
   
   e = b - p = b - y(hat)A
   
   when e is as small as possible then y(hat) is the 
   lease square solution
   
   we will be able to solve Ay(hat) = p
   
   The solution Ay(hat) = p leaves the least possible error "e"
   
   ||(Ay-b) >square> || = || (Ay -p) <square> || + || e <square>||
   
   The Ax -b in the column space is perpendicular to the e 
   in the left null space 
   
   We reduce Ay -p  to zero by choosing y to be y(hat)
   
   The least square solution y (hat) make E = || Ax -b|| <square> as 
   small as possible
   
   e is perpendicular to the column space of A as the nearest
   possible point from b to the column space of A will be the
   perpendicular line and it will cause the least error
   
   A.(b - y(hat)A) = 0
   
   A(transpose). (b - y(hat)A) = 0
   A(transpose)b = A(transpose)y(hat)A
   y(hat) =inv (A(transpose).A)  A(transpose)b 
   
"""  
   
"""
   Lets start the implemntation
 """
   
"""   
  We will need to add a vector of ones to our 
  feature matrix for the intercept term.
"""
   
K = np.ones (shape = X.shape)
print ("K shape {}".format(K.shape))

A = np.concatenate ((K , X), axis = 1)
print ("Matrix A \n {} \n".format(A))



A_Transpose = A.transpose()
print (" A<transpose> \n {} \n".format (A_Transpose))

M = A_Transpose.dot(A)
print  ("A<transpose>.A \n {} \n". format (M))

M_inv = np.linalg.inv (M)
print ("inv (A <transpose>.A) \n {} \n". format (M_inv))


N = A_Transpose.dot (Y)
print ("A<transpose>.b \n {} \n".format (N))


"""
Bringing all together 
"""

coeffs = M_inv.dot (N)
#print ("Coefficients \n {} \n". format (coeffs))

b,m = coeffs

print ("Slope  :  {} \n".format(m))
print ("Intercept : {} \n".format(b))

reg_line = [(m*x) + b for x in IN]

print (reg_line)
reg_line_a = np.asarray(reg_line)


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




   
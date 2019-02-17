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

IN = np.array ([34.0,108.0,64.0,88.0,99.0,51.0])
OUT = np.array ([5.0,17.0,11.0,8.0,14.0,5.0])



corelation= np.corrcoef(IN,OUT)
print ("Pearson corelation coeff {}".format(corelation))


'''
calculate the centrod the regression line should pass through it
'''
IN_mean = np.mean(IN)
print ("Input data mean {}".format (IN_mean))
OUT_mean = np.mean(OUT)
print ("Output data mean {}".format (OUT_mean))



plt.scatter (IN,OUT)
plt.scatter (IN_mean,OUT_mean,  color='red', marker='v')
plt.ylabel ("Dependent Variable")
plt.xlabel ("Independent Variable")
plt.title ("Scatter Plot")
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

reg_line_a = np.asarray(reg_line)
print (reg_line_a)

plt.scatter (IN, OUT, color='red')
plt.scatter (IN_mean,OUT_mean,  color='black', marker='v')
plt.plot (X, reg_line, linestyle='-', marker='o', color='green')
plt.ylabel ("Dependent Variable")
plt.xlabel ("Independent Variable")
plt.title  ("Regression line")
plt.show()


"*********************************************"
"""
Find the SST
"""
error = np.subtract (Y,OUT_mean)
error = np.power(error, 2)
SST = np.sum (error)
print ("SST {} ". format (SST))

"*********************************************"

"*********************************************"
"""
Find the sqare errors 
"""

error = np.subtract (reg_line_a,Y)
error = np.power(error, 2)
SSE = np.sum (error)
print ("SSE {} ". format (SSE))

"*********************************************"

"*********************************************"
"""
Find the SSR
"""

SSR = SST -SSE
print ("SSR {} ". format (SSR))

"*********************************************"

"**********************************************"
"Find Rsquared"

r2 = np.divide (SSR,SST)
print ("Coefficient of determination (r quare)  {}".format(r2))


" Adjusted R square "
r2_bar = 1 - ((1-r2) * ((IN.size-1)/ (IN.size - 2)))

print ("Adjusted R square {} ".format(r2_bar))

"**********************************************"


"**********************************************"

MSE = np.divide (SSE , IN.size -2)
print ("MSE  {}".format(MSE))

"standard error or Root mean square error"
SE = np.sqrt (MSE)
print ("Standard Error (RMSE) {}".format(SE))

"*********************************************"

"*********************************************"

"Confidence interval 95% for the slope "
" b1 +- t[alpha/2]sb1] "

confidence_level = 95
alpha = 1 - (confidence_level/100)
print ("Alpha {}".format(alpha))


"standard deviation of the slope"
error = np.subtract (X,IN_mean)
error = np.power(error, 2)
SSX = np.sum (error)

print (SSX)
sb1 = np.divide (SE, np.sqrt(SSX))

print ("Standard Deviation of the slope {}".format(sb))

CI1 = m + 2.776 * sb1
CI2 = m - 2.776 * sb1

print ("Upper CI (95 %) {} Lower CI (95 %) {}". format(CI1,CI2))

""" 
Its telling we are 95% confident that the interval (0.02885, 0.2636)
contains the true slope of the regression line
"""

"""
Does the interval contain zero . No
So we can reject the NULL hypothesis that the slope is 0

"""

""" Test Ststsitic """
""" t = b1 /sb1"""

t = m / sb1
print ("T statistic {}".format(t))


"""

t vs t critical
3.458 > 2.776 is SIGNIFICANT

"""

"*********************************************"


"*********************************************"

""" Confidence interval of y<hat> """
""" y <hat> +- tsy <hat> """


#for Total bill of $64

y = b + m * 64

sy = SE * (np.sqrt ( (1/IN.size) + (np.power((64 - IN_mean),2) / SSX)))

print (sy)

CI1 = y - 2.776 * sy
CI2 = y + 2.776 * sy

print ("Upper CI (95 %) {} Lower CI (95 %) {}". format(CI1,CI2))


"*********************************************"

"**********************************************"

#prediction interval 

s2_pred = np.power (SE,2) + np.power (sy , 2)
s_pred = np.sqrt (s2_pred)

print ("s_pred {}".format(s_pred))

CI1 = y - 2.776 * s_pred
CI2 = y + 2.776 * s_pred

print ("Upper CI (95 %) {} Lower CI (95 %) {}". format(CI1,CI2))

"********************************************"


"""
Lets verify via sklearn
"""

print (" \n \n \n From sklearn")

regressor = LinearRegression()  
regressor.fit(X, Y)  
print(regressor.intercept_) 
print(regressor.coef_)  




   
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 08:22:40 2019

@author: ajodas
"""

import numpy as np
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import scipy.stats as sp_stats
from prettytable import PrettyTable

print ("Linear Regression parameters");

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


def compute_centroid (x, y) :
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return (x_mean, y_mean)

def plot_regression (x,y, x_mean, y_mean, reg_line):
    #print (reg_line)
    plt.scatter (x,y,color='red')
    plt.scatter (x_mean,y_mean,  color='red', marker='v')
    plt.plot (x, reg_line, linestyle='-', marker='o', color='green')
    plt.ylabel ("DV")
    plt.xlabel ("IV")
    plt.title  ("Regression line")
    plt.show()
    
def construct_regression_line (X, slope , intercept, Y):
    r, c = slope.shape
    reg_line = np.array([((np.sum (np.multiply (slope,x.reshape(r,c)))) + intercept)  for x in X])
    return reg_line    

"""
Total sum of square errors (SST)
-->The SST specifies how well the mean fits the data.
-->The error when there is only the dependent variable 
and the approximation is the mean

"""
def compute_SST (X, Y): 
    mean = np.mean (Y)
    error = np.subtract (X, mean)
    error = np.power(error, 2)
    SST = np.sum (error)
    return SST

"""
Sum of square errors (SSE)
--> SS residual
---> The SSE Sspecifies how well the regression line fits the data
---> SSE is the sum of the squared residuals (errors) a measure of the
variability of the observations about the regression line.
"""
def compute_SSE (X, slope , intercept, Y):
    reg_line = construct_regression_line (X, slope, intercept, Y)
    reg_line =  reg_line[:,np.newaxis]   
    error = np.subtract (reg_line,Y)   
    error = np.power(error, 2)
    SSE = np.sum (error)
    return SSE

"""
Sum of squared due to regression
---> SS model
--> SSR specifies how much better the regression line is compared to the mean 
--> the SSR is large it uses up more SST and therefore SSE is smaller relative to SST
"""
def compute_SSR (SST, SSE):
    SSR = SST - SSE 
    return SSR

"""
The df(Regression) is one less than the number of parameters being estimated. 
There are k predictor variables and so there are k parameters for the 
coefficients on those variables. 
There is always one additional parameter for the constant 
so there are k+1 parameters. 
But the df is one less than the number of parameters, 
so there are k+1 - 1 = k degrees of freedom. 
That is, the df(Regression) = # of predictor variables.

The df(Residual) is the sample size minus the number of parameters 
being estimated, so it becomes df(Residual) = n - (k+1) or 
df(Residual) = n - k - 1. 

The df(Total) is still one less than the sample size as it was before. 
df(Total) = n - 1.

"""
def compute_degress_freedom (X, slope):
    k = len (slope)    
    n = len (X)   
    df_residual = n-k-1
    df_regression = k
    df_total      = n-1
    return (df_residual, df_regression, df_total)

"""
Mean Square Error
--> MSE is an estimate of the variance of the error.
--> How spread out the data points are from the regression line.
--> MSE is SSE / degree of freedom
"""
def compute_mse (SSE, df_residual):
    MSE = np.divide (SSE , df_residual)
    return MSE

"""
Root Mean Square Error or SE
The standard error of the estimate is the standard deviation of the error term.
It is the average distance an observation falls nfrom the regression line
in units of the dependent variable.
"""    
def compute_root_mean_square_error (MSE):
    SE = np.sqrt (MSE)
    return SE

"""
It measures the proportion of the variation in your dependent variable 
explained by all of your independent variables in the model. 
It assumes that every independent variable in the model helps 
to explain variation in the dependent variable. In reality, 
some independent variables (predictors) don't help to explain 
dependent (target) variable. In other words, some variables do not 
contribute in predicting target variable.
 
The toatl sum of squares can be explained by using the estimated 
regression equation to predict the DV . The remainder is the error 
"""
def compute_r_squared (SSR, SST):
    r_square = np.divide (SSR,SST)
    return r_square

"""
Adjusted R2 also indicates how well terms fit a curve or line, 
 but adjusts for the number of terms in a model. If you add more and
 more useless variables to a model, adjusted r-squared will decrease. 
 If you add more useful variables, adjusted r-squared will increase.
 
It measures the proportion of variation explained by only those independent 
variables that really help in explaining the dependent variable. 
It penalizes you for adding independent variable that do not help 
in predicting the dependent variable. 
"""
def compute_adjusted_r_squared (r_squared, df_residual, df_total):
    r_square_bar = 1 - ((1-r_squared) * (df_total/ df_residual))
    return r_square_bar
    

def compute_press_statistic(X, slope, intercept, Y):
    K = np.ones (shape = X.shape)
    A = np.concatenate ((K , X), axis = 1)
    A_Transpose = A.transpose()
    M = A_Transpose.dot(A)
    M_inv = np.linalg.inv (M)
    L1 = A.dot(M_inv)
    hat = L1.dot(A_Transpose)
    den = (1 - np.diagonal(hat))
    reg_line = construct_regression_line (X, slope, intercept, Y)
    reg_line =  reg_line[:,np.newaxis]      
    res = np.subtract (reg_line,Y)     
    den =  den[:,np.newaxis]     
    sqr = np.divide (res,den)   
    sqr = np.square(sqr)    
    return sqr.sum()

def compute_r_square_predicted(X, Y, slope, intercept, SST):    
    PRESS = compute_press_statistic (X, slope , intercept, Y)  
    r_square_pred = 1 - (PRESS / SST )   
    return r_square_pred

"""
The standard error of the regression slope, s (also called the standard 
error of estimate) represents the average distance that your observed values 
deviate from the regression line
"""
def compute_standard_error_predictor (predictor, standard_error):
    SST = compute_SST (predictor,predictor)
    s_predictor = np.divide (standard_error, np.sqrt(SST))
    return s_predictor
    
def compute_confidence_interval_predictor (standard_error_predictor, predictor_val, confidence_level, df):    
    alpha = 1 - (confidence_level/100)
    alpha = alpha/2    
    cv = sp_stats.t.ppf(1.0 - alpha, df)
    CI1 = predictor_val + cv * standard_error_predictor
    CI2 = predictor_val - cv * standard_error_predictor
    return cv,CI1,CI2
        
def compute_tstat_pvalue(predictor_val, standard_error_predictor, df):
    t = predictor_val / standard_error_predictor
    p = (1 - sp_stats.t.cdf(abs(t), df)) * 2
    return t,p
    
def compute_standard_error_intercept (X, standard_error, x_value):
    SSX = compute_SST (X,X)
    sy = standard_error * (np.sqrt ( (1/X.size) + (np.power(x_value - np.mean(X),2) / SSX)))
    return sy    

def compute_s_predicted_value (SE, sy):
    s2_pred = np.power (SE,2) + np.power (sy , 2)
    s_pred = np.sqrt (s2_pred)
    return s_pred

def compute_fstat (r_squared , df1, df2):
    s1 = np.divide (r_squared , df1 )    
    s2 = np.divide ((1 - r_squared) , df2)    
    F = np.divide (s1,s2)
    return F
    
def compute_fstat_pvalue(fvalue, confidence_level, df1, df2):
    alpha = 1 - (confidence_level/100)
    alpha = alpha/2      
    #crit = sp_stats.f.ppf(alpha, df1, df2)  
    crit = sp_stats.f.ppf(q=1-0.05, dfn=df1, dfd=df2)
    p = 1 - sp_stats.f.cdf( fvalue, dfn=df1, dfd=df2) 
    return crit,p
     
# ******************************************************************* #



def linear_regression_compute_parameters (IV, DV, confidence_level):

    t = PrettyTable(['Parameters', 'Value'])
    
    
    corelation= np.corrcoef(IV,DV)
    t.add_row(['Corelation', corelation[0][1]])
    
    t.add_row(['Confidence Level', str (confidence_level) + ('%')])
    
    alpha = 1 - (confidence_level/100)
    alpha = alpha/2
    t.add_row(['Alpha', alpha])    
    
    '''
    calculate the centrod the regression line should pass through it
    '''
    x_mean,y_mean = compute_centroid (IV,DV)
    t.add_row(['Centroid', (str(x_mean) + ',' +str(y_mean))])
    
    X = IV[:,np.newaxis]
    Y = DV[:,np.newaxis]
    
    slope, intercept = compute_coeffs(X,Y)
    
    t.add_row(['Slope', slope[0][0]])
    t.add_row(['Intercept', intercept])
    
    reg_line = construct_regression_line (X, slope, intercept, Y)
    
    plot_regression (IV, DV, x_mean, y_mean, reg_line)
    
    SST = compute_SST (Y, Y)
    t.add_row(['SST', SST])
    
    SSE = compute_SSE (X, slope, intercept, Y)
    t.add_row(['SSE', SSE])
    
    SSR = compute_SSR (SST, SSE)
    t.add_row(['SSR', SSR])
    
    df_residual, df_regression, df_total = compute_degress_freedom (X , slope)

    t.add_row(['DF (Regression)', df_regression])
    t.add_row(['DF (Residual)', df_residual])
    t.add_row(['DF (Total)', df_total])
    
    MSE = compute_mse (SSE, df_residual)
    t.add_row(['MSE', MSE])
    
    SE = compute_root_mean_square_error (MSE)
    t.add_row(['SE', SE])
    
    r_squared = compute_r_squared (SSR, SST)
    t.add_row(['R Squared', r_squared])
    
    r_square_adjusted = compute_adjusted_r_squared (r_squared, df_residual, df_total)
    t.add_row(['R Squared Adjusted', r_square_adjusted])
    
    r_sqaure_predicted = compute_r_square_predicted (X, Y, slope, intercept, SST)
    t.add_row(['R Squared predicted', r_sqaure_predicted])
    
    F_ratio = compute_fstat (r_squared , df_regression, df_residual)
    crit,p = compute_fstat_pvalue (F_ratio, 95, df_regression, df_residual)
    t.add_row(['F ratio/Critical/p-value', str(F_ratio) + "/" + str(crit) + "/" + str(p)])
    
    
    sb1 = compute_standard_error_predictor (X, SE)
    t.add_row(['Standard Error of the predictor b1',sb1])
    
    cv, CI1,CI2 = compute_confidence_interval_predictor (sb1, slope, 95, df_residual)
    t.add_row(['CV CI1 CI2',(str(cv) + ',' + str(CI1) + ',' + str(CI2))])
    
    t_stat,p = compute_tstat_pvalue (slope, sb1, df_residual)
    t.add_row(['T statistics P value',(str(t_stat) +',' + str(p))])
    
    sy = compute_standard_error_intercept (X, SE, 0)
    t.add_row(['Standard Error of the predictor b0',sy])
    
    t_stat,p = compute_tstat_pvalue (intercept, sy, df_residual)
    t.add_row(['T statistics P value',(str(t_stat) +',' + str(p))])
    
    cv, CI1,CI2 = compute_confidence_interval_predictor (sy, intercept, 95, df_residual)
    
    t.add_row(['CV CI1 CI2',(str(cv) + ',' + str(CI1) + ',' + str(CI2))])

    print (t)
    return (slope,intercept,df_residual,df_regression,df_total, SE)


def compute_CI_PI(X, intercept, df_residual, SE):
    t = PrettyTable(['Parameters', 'Value'])
    #for Total bill of $64
    t.add_row(['Total Bill Amount',64])
    y = slope * 64 + intercept
    sy = compute_standard_error_intercept (X, SE, 64)
    t.add_row(['Standard Error of the predictor b0',sy])

    cv,CI1,CI2 = compute_confidence_interval_predictor (sy, y, 95, df_residual)
    t.add_row(['CV CI1 CI2 (Confidence Interval)',(str(cv) + ',' + str(CI1) + ',' + str(CI2))])

    spred = compute_s_predicted_value (SE, sy)
    cv,CI1,CI2 = compute_confidence_interval_predictor (spred, y, 95, df_residual)
    t.add_row(['CV CI1 CI2 (Prediction Interval)',(str(cv) + ',' + str(CI1) + ',' + str(CI2))])
    print (t)
    
def compute_linear_regression_params_sklearn(IV, DV): 
    t = PrettyTable(['Parameters', 'Value'])
    t.add_row(['sklearn',""])
    X = IV[:,np.newaxis]
    Y = DV[:,np.newaxis]
    regressor = LinearRegression()  
    regressor.fit(X, Y)  
    t.add_row(['Intercept',regressor.intercept_])
    t.add_row(['slope',regressor.coef_])
    print (t)


if __name__== "__main__":
    IV = np.array ([34.0,108.0,64.0,88.0,99.0,51.0])
    DV = np.array ([5.0,17.0,11.0,8.0,14.0,5.0])
    
    confidence_level = 95
    
    slope, intercept, df_residual, df_regression, df_total, SE, = linear_regression_compute_parameters (IV,  DV, 95)
    
    X = IV[:,np.newaxis]
    compute_CI_PI (X, intercept, df_residual, SE)
    
    compute_linear_regression_params_sklearn(IV, DV)


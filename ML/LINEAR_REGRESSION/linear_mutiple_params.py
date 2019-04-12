#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:52:47 2019

@author: ajodas
"""



import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp_stats
import linear_reg_params
import linear_regression_library as lreg_lib
import linear_reg_params as lreg_param

from prettytable import PrettyTable

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

def compute_multiple_regression (X, Y, confidence_level):
    
    t = PrettyTable(['Parameters', 'Value'])    
    
    slope,intercept = lreg_lib.compute_coeffs (X , Y)
    t.add_row(['Slope', slope])
    t.add_row(['Intercept', intercept])
    
    SST = lreg_lib.compute_SST (Y ,Y)
    t.add_row(['SST', SST])
    
    SSE = lreg_lib.compute_SSE (X, slope , intercept, Y)
    t.add_row(['SSE', SSE])

    SSR = lreg_lib.compute_SSR (SST, SSE)
    t.add_row(['SSR', SSR])
    
    df_residual, df_regression, df_total = lreg_lib.compute_degress_freedom (X, slope)
    
    t.add_row(['DF (Regression)', df_regression])
    t.add_row(['DF (Residual)', df_residual])
    t.add_row(['DF (Total)', df_total])
    
    MSE = lreg_lib.compute_mse (SSE, df_residual)
    t.add_row(['MSE', MSE])
    
    SE = lreg_lib.compute_root_mean_square_error (MSE)
    t.add_row(['SE', SE])
    
    r_squared = lreg_lib.compute_r_squared (SSR,SST)
    t.add_row(['R Squared', r_squared])
    
    r_square_adjusted = lreg_lib.compute_adjusted_r_squared (r_squared, df_residual, df_total)
    t.add_row(['R Squared Adjusted', r_square_adjusted])
    
    r_sqaure_predicted = lreg_lib.compute_r_square_predicted (X, Y, slope, intercept, SST)
    t.add_row(['R Squared predicted', r_sqaure_predicted])
    
    F_ratio = lreg_lib.compute_fstat (r_squared , df_regression, df_residual)
        
    crit,p = lreg_lib.compute_fstat_pvalue (F_ratio, 95, df_regression, df_residual)
    t.add_row(['F ratio/Critical/p-value', str(F_ratio) + "/" + str(crit) + "/" + str(p)])
    
    results = lreg_lib.compute_adj_SS (X, Y, SSE)
    t.add_row(['Adjusted SS', results])
    
    results = lreg_lib.compute_adj_MS (results, 1)
    t.add_row(['Adjusted MS', results])
    
    for i in range(results.size) :
        f_value = lreg_lib.compute_fstat_individual(results[i], MSE)
        crit,p = lreg_lib.compute_fstat_pvalue (f_value, 95, 1, df_residual)
        t.add_row(['F ratio/Critical/p-value', str(f_value) + "/" + str(crit) + "/" + str(p)])
    
    #f_value = linear_reg_params.compute_fstat_individual(results[1], MSE)
    #crit,p = linear_reg_params.compute_fstat_pvalue (f_value, 95, 1, df_residual)
    #t.add_row(['F ratio/Critical/p-value', str(f_value) + "/" + str(crit) + "/" + str(p)])

    sb = lreg_lib.compute_standard_deviation_predictor (X, MSE)
    t.add_row(['Standard Deviation Predictor', sb])
    
    tvalue = lreg_lib.compute_predictor_t_val (sb, slope, intercept)
    t.add_row(['T value', tvalue])
    
    p_value = lreg_lib.compute_predictor_p_val (tvalue, df_residual)
    t.add_row(['P value', p_value])
    
    vif = lreg_lib.compute_vif (X)
    t.add_row(['VIF', vif])
    
    print (t)
    

#Independent variable
MilesTravelled = np.array ([89,66,78,111,44,77,80,66,109,76])
NumDeliveries = np.array ([4,1,3,6,1,3,3,2,5,3])
GasPrice = np.array ([3.84,3.19,3.78,3.89,3.57,3.57,3.03,3.51,3.54,3.25])

#Dependent Variable
TravelTime = np.array ([7,5.4,6.6,7.4,4.8,6.4,7,5.6,7.3,6.4])

df = TravelTime.size

plot_scatter (MilesTravelled, TravelTime, "Miles Travelled", "Travel Time")
plot_scatter (NumDeliveries, TravelTime, "Number of Deliveries", "Travel Time")
plot_scatter (GasPrice, TravelTime, "Gas Price", "Travel Time")

plot_scatter (MilesTravelled, NumDeliveries, "Miles Travelled", "Number of Deliveries")
plot_scatter (NumDeliveries, GasPrice, "Number of Deliveries", "Gas Price")
plot_scatter (GasPrice, MilesTravelled, "Gas Price", "Miles Travelled")

corelation , p= compute_cor_p_value (MilesTravelled,TravelTime,df)
print ("Pearson corelation coeff miles travlled vs travel time {}".format(corelation[0][1]))

corelation, p= compute_cor_p_value (NumDeliveries,TravelTime,df)
print ("Pearson corelation coeff num deliveries vs travel time {}".format(corelation[0][1]))

corelation , p= compute_cor_p_value (GasPrice,TravelTime,df)
print ("Pearson corelation coeff gas price vs travel time {}".format(corelation[0][1]))

corelation , p= compute_cor_p_value (MilesTravelled,NumDeliveries,df)
print ("Pearson corelation coeff miles travlled vs num deliveries {}".format(corelation[0][1]))

corelation , p= compute_cor_p_value (NumDeliveries,GasPrice,df)
print ("Pearson corelation coeff num deliveries vs gas price {}".format(corelation[0][1]))

corelation , p= compute_cor_p_value (MilesTravelled,GasPrice,df)
print ("Pearson corelation coeff miles travlled vs gas price {}\n".format(corelation[0][1]))


print ("MilesTravelled (IV) vs TravelTime (DV)")
lreg_param.linear_regression_compute_parameters (MilesTravelled, TravelTime, 95)

print ("NumDeliveries (IV) vs TravelTime (DV)")
lreg_param.linear_regression_compute_parameters (NumDeliveries, TravelTime, 95)

print ("GasPrice (IV) vs TravelTime (DV)")
lreg_param.linear_regression_compute_parameters (GasPrice, TravelTime, 95)

MilesTravelled = MilesTravelled[:,np.newaxis]
NumDeliveries = NumDeliveries[:,np.newaxis]
GasPrice = GasPrice[:,np.newaxis]
TravelTime =  TravelTime[:,np.newaxis]

print ("***********************************")
print ("MilesTravelled (IV) NumDeliveries (IV) vs TravelTime (DV)")

X12 = np.hstack ([MilesTravelled ,NumDeliveries])
compute_multiple_regression (X12 , TravelTime, 95)

print ("***********************************")
print ("MilesTravelled (IV) GasPrice (IV) vs TravelTime (DV)")
X13 = np.hstack ([MilesTravelled ,GasPrice])
compute_multiple_regression (X13 , TravelTime, 95)

print ("***********************************")
print ("NumDeliveries (IV) GasPrice (IV) vs TravelTime (DV)")
X23 = np.hstack ([NumDeliveries ,GasPrice])
compute_multiple_regression (X23 , TravelTime, 95)

print ("***********************************")

print ("***********************************")
print ("MilesTravelled vs NumDeliveries (IV) GasPrice (IV) vs TravelTime (DV)")
X123 = np.hstack ([MilesTravelled, NumDeliveries ,GasPrice])
compute_multiple_regression (X123 , TravelTime, 95)











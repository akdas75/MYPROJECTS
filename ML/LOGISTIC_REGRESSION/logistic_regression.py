#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:13:44 2019

@author: ajodas
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid (theta, X):
    #print ("Theta {}".format(theta))
    #print ("X {}".format(X))
    z = np.dot(X, theta)
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid

def loss (h, y):
    return (-y * np.log (h) - (1 - y) * np.log(1 - h)).mean()
    

def gradient_descent (X, theta, Y, alpha):
    
    h =  sigmoid (theta, X)
    #print (h)
    #print (h.shape)
    #print (Y.shape)
    XT = X.transpose()
    gradient = np.dot(XT, (h - Y)) / Y.shape[0]
    l = loss (h, Y)
    #print ("Loss {}".format (l))
    #print ("Loss {}".format(h-Y))
    #print ("Gardient {}".format(gradient))
    theta -= alpha * gradient
    
    return theta,l


    

def logistic_regression (X, Y, num_iters, learning_rate):
    print ("X shape {}".format (X.shape))
    Z = np.ones((X.shape[0],1))
    X_mod = np.concatenate ((Z , X) , axis =1)
    #print (X_mod)
    
    n = X_mod.shape[1]
    print ("Number of feature vectors {} \n".format (n-1))

    # initializing the parameter vector...
    theta = np.zeros(n)
    
    theta = theta [:, np.newaxis]
    Y = Y [:, np.newaxis]
    print (theta.shape)
    #hypothesis calulation
    
    loss = []
    for i in range (0 , num_iters) :
        theta, l = gradient_descent (X_mod, theta, Y, learning_rate)
        loss.append (l)
    print (theta)    

    n_iterations = [x for x in range(0,num_iters)]
    #print (loss)
    plt.plot (loss, n_iterations)
    plt.show()
    
    return theta

" ******************************************************************* "

data = np.loadtxt ('dataset.txt', delimiter = ',')
#print (data)

print (data.shape)

X_train = data [0:, [0,1]] #feature set
Y_train = data [0:, -1]     #label

#print (X_train)
#print (Y_train)

x0 = np.ones((np.array([x for x in Y_train if x == 0]).shape[0], X_train.shape[1]))

x1 = np.ones ((np.array ([x for x in Y_train if x == 1]).shape[0], X_train.shape[1]))

"""
x0 and x1 are matrices containing +ve and -ve examples from the
dataset, initialized to 1
"""

k0 = k1 = 0
#print (len(Y_train))
for i in range (0 , len(Y_train)):
    if Y_train [i] == 0:
        x0[k0] = X_train[i]
        k0 = k0 + 1
    else:
       x1[k1] = X_train[i]
       k1 = k1 + 1
       
X = [x0, x1]
color = ['blue' , 'red']

#Z = zip (X , color)
#print (list(Z))

for x , c in zip (X , color):
    #print (x)
    if c == "green":
        plt.scatter(x[:,0],x[:,1],color = c,label = "Not Admitted")
    else:
        plt.scatter(x[:,0], x[:,1], color = c, label = "Admitted")
        
plt.xlabel("Marks obtained in 1st Exam")
plt.ylabel("Marks obtained in 2nd Exam")
plt.legend()
plt.show()

num_iterations = 100000
theta = logistic_regression (X_train, Y_train, num_iterations, 0.001)
X_mod = np.concatenate ((np.ones((X_train.shape[0],1)) , X_train) , axis =1)

h = sigmoid (theta ,X_mod)

#Taking 0.5 as threshold
for i in range(0, h.shape[0]):
    if h[i] > 0.5:
        h[i] = 1
    else:
        h[i] = 0
k = 0
for i in range(0, h.shape[0]):
    if h[i] == Y_train[i]:
        k = k + 1
accuracy = k/Y_train.shape[0]
print ("Accuaracy {}".format(accuracy))



tp = fp = tn = 0
# tp -> True Positive, fp -> False Positive
for i in range(0, h.shape[0]):
    if h[i] == Y_train[i] == 0:
        tp = tp + 1
    elif h[i] == 0 and Y_train[i] == 1:
        fp = fp + 1
    elif h[i] == Y_train[i] == 1:
        tn = tn + 1
        
print ("True Positive {}".format(tp)) 
print ("True Negative {}".format(tn))  
print ("False Positive {}".format(fp))       
precision = tp/(tp + fp)

fn = 0
# fn -> False Negatives
for i in range(0, h.shape[0]):
    if h[i] == 1 and Y_train[i] == 0:
        fn = fn + 1
recall = tp/(tp + fn)
print ("False Negative {}".format(fn))

print ("Precision {}".format(precision)) 
print ("Recall {}".format(recall)) 

f1_score = (2 * precision * recall)/(precision + recall)
print ("F1 score {}".format(f1_score))






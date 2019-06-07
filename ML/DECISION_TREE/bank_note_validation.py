#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 04:37:31 2019

@author: ajodas
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def compute_gini_index(list_mem):
    total = sum (list_mem)
    length = len(list_mem)
    gini = 0
    
    for i in range (0 , length):
        val = list_mem[i]
        if val == 0:
            continue
        p_num = val/total
        p_num_sq = np.square(p_num)
        gini += p_num_sq 
        
    gini = 1 - gini    
        
    return  gini

def get_split_df (df, feature_name, split_value):  
    sub_features_left = df[df[feature_name] < split_value]
    sub_features_right = df[df[feature_name] >= split_value]
    return sub_features_left,sub_features_right 

def compute_feature_gini_index (df, feature_name, split_value, target_name):
   
    total_gini = 0
    
    sub_features_left , sub_features_right = get_split_df (df, feature_name, split_value)
    #print (sub_features_1)
    #print ("******************************")
    #print (sub_features_2)
    #print ("#############################")
        
    clss_freq = []
    for j in range (0 , len (df[target_name].unique())):
        clss_freq.append ((sub_features_left [sub_features_left[target_name] == df[target_name].unique()[j]]).shape[0])   
        
    left_part_gini = compute_gini_index (clss_freq)
            
    clss_freq = []
    for j in range (0 , len (df[target_name].unique())):
        clss_freq.append ((sub_features_right [sub_features_right[target_name] == df[target_name].unique()[j]]).shape[0])
            
    right_part_gini = compute_gini_index (clss_freq)
          
    total_gini = (sub_features_left.shape[0]/df.shape[0]) * left_part_gini
    total_gini += (sub_features_right.shape[0]/df.shape[0]) * right_part_gini
    
    #print (' {} :  {} : {} '.format (feature_name , split_value, total_gini))
        
    return (total_gini , feature_name, split_value , sub_features_left, sub_features_right)


def compute_dataframe_gini_index (df, target_name):
     
     best_gini =100     
    
     # loop for num of columns in the dataframe apart from target
     for i in range (0, df.shape[1] - 1):
            # For each column loop in for the num rows 
            for j in range (0, (df.shape[0])):           
                gini , feature_name , feature_val , sub_feature_left , sub_feature_right =  \
                compute_feature_gini_index (df, df.columns[i], df.iloc[j][i],target_name)
            
                #print (best_gini)
                if gini < best_gini:
                    best_gini           = gini;                 
                    best_feature        = feature_name
                    best_val            = feature_val
                    split_feature_left  =  sub_feature_left
                    split_feature_right =  sub_feature_right
                    index               = i 
     
     #print (' BEST {} :  {} : {} : {} '.format (index , best_feature, best_val, best_gini))   
     return ({'index' : index ,'groups' : [split_feature_left , split_feature_right] , 'value' : best_val}) 
 
# Create a terminal node value
def to_terminal(df , target_name):   
    val_cnt = df[target_name]. value_counts().values
    unq = df[target_name].unique()    
    slected_class = unq [np.argmax(val_cnt)]    
    return slected_class

def split (node, max_depth, min_size, depth , target_name):   
    
    left, right = node['groups']   
    
    del(node['groups'])
    
    if left.empty or right.empty :        
        node ['left'] = node ['right'] = to_terminal (pd.concat((left ,right)), target_name)
        return
        
    if depth >= max_depth :         
        node ['left'], node ['right'] = to_terminal (left ,target_name) , to_terminal (right, target_name)
        return
    
    #process left child
    if (left.shape[0] <= min_size):
        node ['left'] =  to_terminal (left , target_name)
    else:    
        node ['left'] = compute_dataframe_gini_index(left , target_name)
        split (node ['left'] , max_depth, min_size, depth+1 , target_name)
    
    #process right child
    if (right.shape[0] <= min_size):
        node ['right'] =  to_terminal (right , target_name)
    else:    
        node ['right'] = compute_dataframe_gini_index(right , target_name)
        split (node ['right'] , max_depth, min_size, depth+1 , target_name)  
        
def predict(node, row):    
    if row[node['index']] < node['value']:       
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:        
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']     
        
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0        
        
# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))


def build_tree (train, max_depth, min_size ,target_name):
    root = compute_dataframe_gini_index (train, target_name)    
    split(root, max_depth, min_size, 1, target_name)
    #print_tree (root , max_depth)
    return root

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size , 'class')
    #print (tree)
    predictions = list()
    for index, row in test.iterrows():
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)
    

# **************************************************************************
'''    
test_data = pd.read_csv('test_data.csv')
test_data    

build_tree (test_data, 3, 1, 'Y')
'''
# *************************************************************************
'''        
test_data = pd.read_csv('test_data.csv')
print (test_data)

#  predict with a stump
stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
for index, row in test_data.iterrows():
    #print (row)
    prediction = predict(stump, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
'''    
# ************************************************************************   

bnote = pd.read_csv('BankNote.csv' , header = None)
bnote.columns = ['variance' , 'skewness' , 'kurtosis' , 'entropy' , 'class']

print (bnote.head())
print ('Data shape {}'.format(bnote.shape)) 

X = bnote
y = bnote['class']

kf = KFold(n_splits=5)
kf.get_n_splits (X)

print (kf)

max_depth = 5
min_size = 10
for train_index,test_index in kf.split (X):
    print ('Iter Train {} Test {}'.format(X_train.shape , X_test.shape))
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    predictions = decision_tree (X_train , X_test, max_depth, min_size)
    
    
    metric = accuracy_metric (predictions , y_test.values)
    print (metric)

print ('Done')    

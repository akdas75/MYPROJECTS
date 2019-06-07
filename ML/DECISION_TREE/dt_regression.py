#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:27:39 2019

@author: ajodas
"""

import pandas as pd
import numpy as np

def compute_standard_deviation(list_mem):
    
    n = len (list_mem)
    mean = np.mean (list_mem)
    std = np.std(list_mem)
    CV = (std/mean) * 100   
    return  mean , std, CV 

def compute_target_std (df, target_name):   
    dist = df['Hours Played'].values
    #print (dist)
    mean , std, CV = compute_standard_deviation(dist)
    return mean, std, CV

def get_sub_class_df (df, feature_name):
    sub_features = df[feature_name].unique()
    #print (sub_features)
    
    sub_df = {}
    for i in range (0 , len(sub_features)):
        #print (sub_features[i])
        k = df[feature_name] == sub_features[i]
        #print (df[k])
        sub_df[sub_features[i]] = df[k]
    
    return sub_df , sub_features 

def compute_feature_std (df, feature_name, target_name):
   
    #print (feature_name)
    df1 = df
    
    subcalss_df , sub_features = get_sub_class_df (df1, feature_name)
    #print (subcalss_df)
    #print (sub_features)
    target_class = df1[target_name].unique()    
    
    total_std = 0
    for i in range (0, len(sub_features)):
        df2 = subcalss_df[sub_features[i]]
        #print (df2)
        dist= df2['Hours Played'].values        
            
        #print (dist) 
        mean , std, CV = compute_standard_deviation(dist)
        #print (entropy)
        val = df2.shape[0]/df1[target_name].shape[0]
        #print (val)
        std *= val
        
        #print (df2.shape[0])
        #print (df[target_name].shape[0])
        
        total_std += std
        
    return (total_std)

#computes how much variance is reduced
def compute_IG (std_parent, std):
    return std_parent - std

def compute_split_feature(df , target_col):
    best_std = 0;
    tot_std = 0
    class_name = None
    
    target_mean , target_std , target_CV = compute_target_std (df , target_col)
       
    for i in range (0, len(df.columns) - 1):
        std  = compute_feature_std (df , df.columns[i], 'Hours Played')    
        IG = compute_IG (target_std , std)
        #print ("Feature -- {} ::: std {} IG {}".format(df.columns[i], std, IG))
    
        if (best_std == 0):
            best_std = IG
            split_feature = df.columns[i]
        else :
            if (IG > best_std):
                best_std = IG                
                split_feature = df.columns[i]
        tot_std += std        
                
    if (tot_std == 0.0):
        class_name = df['Hours Played'].unique()
        split_feature = None
        
    #print ("Best IG {} split column :: {}". format(best_std, split_feature))
    return split_feature ,class_name

def compute_branch_split (df , target_col):
    #print (df)
    target_mean , target_std , target_CV = compute_target_std (df , target_col)
    
    
    #print ('{} {} {}'.format(target_mean,target_std,target_CV))
    if target_CV < 8 or df.shape[0] < 3:
        print ("Avg Hours played {}:".format(target_mean))
        return 
    
    root_node, class_name = compute_split_feature(df , target_col)
    
    #if root_node == None:
    #    print (class_name)
    #    return class_name
    
    print ("************************************")
    print ("Root Node :: {}".format(root_node))
    print ("************************************")

    # split the df with respect to the root node 
    unq = df [root_node].unique()
    for i in range (0, len(unq)):
        df_sub = df[df[root_node] == unq[i]]
        print (unq[i])    
        df_sub.drop (root_node , axis = 1, inplace = True)
        next_node = compute_branch_split (df_sub , target_col)
        print ("************************************")
        print ("Next Node ::{} ---> {} ".format(root_node, unq[i]))
        print ("************************************")
    
    return root_node

dict = {'Outlook':['Rainy','Rainy','Overcast','Sunny','Sunny','Sunny','Overcast','Rainy','Rainy','Sunny','Rainy','Overcast','Overcast','Sunny'],
       
        'Temp':['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild' ],
        
        'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
        
        'Windy' : ['False' , 'True', 'False','False','False','True','True','False','False','False','True','True','False','True'],
       
        'Hours Played' : [25, 30, 46, 45, 52, 23, 43, 35, 38, 46, 48, 52, 44, 30 ]}

df = pd.DataFrame.from_dict (dict)
print (df)

target_col = 'Hours Played'

df_sub  = compute_branch_split(df, target_col)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:04:31 2019

@author: ajodas
"""

import pandas as pd
import numpy as np

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


def compute_target_gini_index (df, target_name):
    class_count = df[target_name].value_counts()
    unq = df[target_name].unique()
    
    dist = []
    for i in range (0, len(unq)):
       dist.append (class_count[unq[i]])
    print (dist)    
    gini = compute_gini_index(dist)
    return gini

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

def compute_feature_gini_index (df, feature_name, target_name):
    
    #df1 = df.set_index (feature_name)
    #print (df1)
    
    df1 = df
    
    subcalss_df , sub_features = get_sub_class_df (df1, feature_name)
    #print (subcalss_df)
    target_class = df1[target_name].unique()
    
    
    total_gini = 0
    for i in range (0, len(sub_features)):
        df2 = subcalss_df[sub_features[i]]
        dist=[]
        #print (sub_features[i])
        for j in range (0, len(target_class)):
            #print (target_class[j])
            df3 = df2[df2[target_name] == target_class[j]]
            #print (df3)
            dist.append (df3.shape[0])
            
            
        #print (dist) 
        entropy = compute_gini_index(dist)
        #print (entropy)
        val = df2.shape[0]/df1[target_name].shape[0]
        #print (val)
        entropy *= val
        
        #print (df2.shape[0])
        #print (df[target_name].shape[0])
        
        total_gini += entropy
        
    return (total_gini)

def compute_split_feature(df):  
    class_name = None
    
    gini = 100
    best_gini = 100
    tot_gini  = 0
    split_feature = None
    for i in range (0, len(df.columns) - 1):
        gini = compute_feature_gini_index (df , df.columns[i], 'Play')    
    
        print ("Feature -- {} ::: Gini {}".format(df.columns[i], gini))
    
        if (gini == 0):
            best_gini = gini
            split_feature = df.columns[i]
        else :
            if (gini < best_gini):
                best_gini = gini                
                split_feature = df.columns[i]
        tot_gini += gini        
                

    if (tot_gini == 0.0):
        class_name = df.Play.unique()
        split_feature = None
        
        
    print ("Best Gini {} split column :: {}". format(best_gini, split_feature))
    return split_feature ,class_name

def compute_branch_split (df , target_col):
    print (df)
    target_entropy = compute_target_gini_index (df , target_col)
    print (target_entropy)
    
    root_node, class_name = compute_split_feature(df)
    
    if root_node == None:
        print (class_name)
        return class_name
    
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
        print ("Next Node ::{} ---> {} ---> {}".format(root_node, unq[i], next_node))
        print ("************************************")
    
    return root_node



dict = {'Outlook':['Rainy','Rainy','Overcast','Sunny','Sunny','Sunny','Overcast','Rainy','Rainy','Sunny','Rainy','Overcast','Overcast','Sunny'],
       
        'Temp':['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild' ],
        
        'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
        
        'Windy' : ['False' , 'True', 'False','False','False','True','True','False','False','False','True','True','False','True'],
       
        'Play' : ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No', ]}


df = pd.DataFrame.from_dict (dict)

print (df)

target_col = 'Play'
target_entropy = compute_target_gini_index (df , target_col)
print (target_entropy)

target_col = 'Play'

df_sub  = compute_branch_split(df, target_col)
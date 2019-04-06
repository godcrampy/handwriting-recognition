# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:47:49 2019

@author: SAHIL
"""
#Instructions: Keep this file in the same directory as the 101_ObjectCategories
# Running this file will create a new directory for training data
# Rename the 101_ObjectCategories directory as test_dataset


# Make a Dictionary of all categories
import os
import math
object_list = next(os.walk('test_dataset'))[1]
object_index = {}
for n, category in enumerate(object_list):
    object_index[n] = category


# Split the directory into test and train datasets
    #Create Empty Train Directory
    for object_name in object_index.values():
        os.makedirs("./train_dataset/" + object_name)
    #Move 80% of images to the train_dataset directory
    train_ratio = 0.8
    for i, object_name in object_index.items():
        object_path = './101_ObjectCategories/' + object_index[i]
        file_list = next(os.walk(object_path))[2]
        train_split = math.floor(len(file_list)*train_ratio)
        test_split = math.floor(len(file_list)*(1-train_ratio)) + 1
        
        for k in range(train_split):
            os.rename(object_path + '/' + file_list[k], "./train_dataset/" + object_name + "/" + file_list[k])
            
            
            

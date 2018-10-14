# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:37:28 2018
File to obtain mean and standard deviation of sound files
@author: simon.suthers
"""

#%%

import numpy as np

#%%
#Import DataGenerator2 class in Datagenerator folder
import sys
sys.path.append('../Datagenerator')

from datagenerator2 import DataGenerator2

#%% Unit test
#Set console working directory
            
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

#%% Get mean and standard deviation of record set
 
#list of training data files  
#pickle files should be saved in the same directory as this script
training_files = ['../Data/train' + str(i) + '.pkl' for i in range(1, 8)]

#Generate data
data_generator = DataGenerator2(training_files)

#Get total number of samples
total_samples = data_generator.total_samples

#Generate all data
data = data_generator.gen_batch(data_generator.total_samples)

#Convert data into array of samples
data_set = np.concatenate([item['SampleStd'] for item in data], axis=0)

#Expand data type to avoid overflow error
data_set = np.array(data_set, dtype=np.float64)

#Get mean of samples
mean = np.mean(data_set)

#Get Standard deviation of samples
std = np.std(data_set)












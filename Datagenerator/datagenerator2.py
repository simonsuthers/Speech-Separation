

import numpy as np
import pickle

#%%
#This class returns a list of dict objects. THe dict objects have the form:
#{spectrogram, VAD of bool for each point in spectrogram, 2xIBM for each point in spectrogram}
#{ array of floats(129,100), array of bools(129,100), array of 2 bools(129,100)}

#The class appends pickle files together to create one set. It then takes batches from the set.

#%% Class to combine pickle files and return a batch

class DataGenerator2(object):
    
    #Constructor
    def __init__(self, pkl_list):
        '''pkl_list: .pkl files containing the data set'''
        self.ind = 0  # index of current reading position
        self.samples = []
        self.epoch = 0
        
        # read in all the .pkl files
        for pkl in pkl_list:
            self.samples.extend(pickle.load(open(pkl, 'rb')))
          
        #property for the length of the data set
        self.total_samples = len(self.samples)

        print("%d samples in dataset"%(self.total_samples))
        

    # method to generate the next batch of data
    def gen_batch(self, batch_size):
        # generate a batch of data
        #Get beginning index of data to return
        n_begin = self.ind
        #Get end index of data to return
        n_end = self.ind + batch_size
        
        #if the end is beyond the number of samples
        if n_end >= self.total_samples:
            # reset the index back to the beginning of the dataset
            self.ind = 0
            #Get beginning and end of new dataset
            n_begin = self.ind
            n_end = self.ind + batch_size
            
        self.ind += batch_size
        return self.samples[n_begin:n_end]
     
    #method for the total number of batches from the data (use floor division to get floor integer)
    def total_batches(self, batch_size):
        return len(self.samples) // batch_size
    
    
    #method for the total number of datapoints
    def total_number_of_datapoints(self):
        return sum([item['Sample'].shape[0] for item in self.samples])
    
    
    def restart_batch(self):
        # reset the index back to the beginning of the dataset
        self.ind = 0
        #randomly sort the data again
        np.random.shuffle(self.samples)
        
 
#%% Unit test
#Set console working directory
            
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

#%% test routine to see data
 
#list of training data files  
#pickle files should be saved in the same directory as this script
training_files = ['../Data/train' + str(i) + '.pkl' for i in range(1, 2)]

#Set batch_size. This is the size of data points to be created
batch_size = 64

#Generate data
data_generator = DataGenerator2(training_files)

#Generate a batch of data
data = data_generator.gen_batch(batch_size)

#Sort data by id
data = sorted(data, key=lambda k: k['Id']) 

#Get total number of data points
total_points = data_generator.total_number_of_datapoints()

#Get total number of samples in data set
total_samples = data_generator.total_samples

#%%

del training_files, data_generator, batch_size, data, total_points, total_samples



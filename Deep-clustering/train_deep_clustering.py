'''
Script to train bi-directional RNN model with deep clustering loss function
'''


import tensorflow as tf
import numpy as np


#%% Create dataset

#Import DataGenerator2 class in Datagenerator folder
import sys
sys.path.append('../Datagenerator')

from datagenerator2 import DataGenerator2

#list of training data files   
pkl_list = ['../Data/train' + str(i) + '.pkl' for i in range(1, 4)]

# generator for training set and validation set
data_generator = DataGenerator2(pkl_list)

#Print message to give data about how many batches are in the data
print("Training set: Data samples: %d, Total number of points: %d"%(data_generator.total_samples, data_generator.total_number_of_datapoints()))

#remove variables
del pkl_list

#%% Get data

# Get batch size
batch_size = 64;

#Get total number of batches
total_batches = int(data_generator.total_batches(batch_size))

#Get total number of data points
total_number_of_datapoints = data_generator.total_number_of_datapoints()

# Get sample data
data = data_generator.gen_batch(batch_size);
# Reshape training data
# concatenate training samples together to get 101900 x 129 array
tr_features = np.concatenate([item['Sample'] for item in data], axis=0)
# concatenate labels together to get 6400 x 129 array
Y_labels = (np.concatenate([np.asarray(item['Target'])[:,:,0] for item in data], axis=0)).astype(int)

#Same as neff
n_features = np.shape(tr_features)[1]
#Same as neff
n_classes = np.shape(Y_labels)[1]

del data, tr_features, Y_labels

#%% Get testing set

#list of training data files   
validation_list = ['../Data/test' + str(i) + '.pkl' for i in range(1, 2)]

# generator for training set and validation set
validation_data_generator = DataGenerator2(validation_list)

#Get total number of batches
validation_batches = int(validation_data_generator.total_batches(batch_size))

#Print message to give data about how many batches are in the data
print("Training set: Data samples: %d, Total number of points: %d"%(validation_data_generator.total_samples, validation_data_generator.total_number_of_datapoints()))

#Remove unwanted variables
del validation_list

#%% Reset default graph

tf.reset_default_graph()

#%% Set training parameters
#https://stats.stackexchange.com/questions/330176/what-is-the-output-of-a-tf-nn-dynamic-rnn

training_epochs = 50
n_neurons_in_h1 = 300
n_neurons_in_h2 = 300
learning_rate = 0.01
frames_per_sample = 100

# size embedding dimention
embedding_dimension = 3

#%% Placeholders for inputs, outputs, and Voice Activity detection matrix

#X = tf.placeholder(tf.float32, [batch_size, frames_per_sample, n_features], name="features")
X = tf.placeholder(tf.float32, shape=[None, frames_per_sample, n_features], name="features")

#Y = tf.placeholder(tf.float32, shape=[(batch_size * frames_per_sample), n_classes], name="labels")
Y = tf.placeholder(tf.float32, shape=[(batch_size * frames_per_sample), n_classes, 2], name="labels")


#%% Layer 1

#Create lstm cell with n hidden neurons
with tf.variable_scope('BLSTM1') as scope: 
    lstm_fw_cell1 = tf.contrib.rnn.BasicLSTMCell(n_neurons_in_h1)           
    lstm_bw_cell1 = tf.contrib.rnn.BasicLSTMCell(n_neurons_in_h1) 
    
    #Create first layer
    #Produces 2 lots of 64x100x300 array (1 for forward and 1 for backward)
    outputs1, states1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell1, cell_bw=lstm_bw_cell1, inputs=X, dtype=tf.float32)
    #Produces a 64x100x600 output layer
    state_concate1 = tf.concat(outputs1, 2)

#%% Layer 2
with tf.variable_scope('BLSTM2') as scope: 
    lstm_fw_cell2 = tf.contrib.rnn.BasicLSTMCell(n_neurons_in_h2)
    lstm_bw_cell2 = tf.contrib.rnn.BasicLSTMCell(n_neurons_in_h2)
      
    #Create second layer
    #Produces 2 lots of 64x100x300 array (1 for forward and 1 for backward)
    outputs2, states2 = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell2, cell_bw=lstm_bw_cell2, inputs=state_concate1, dtype=tf.float32)
    #Produces a 64x100x600 output layer
    state_concate2 = tf.concat(outputs2, 2)
    #Produces a 6400x600 output layer
    out_concate = tf.reshape(state_concate2, [-1, n_neurons_in_h2 * 2])


#%% ouput layer

#feed output of bi-driectional layer into feed forward layer
#layer with 600 (2 * n_neurons_in_h2) inputs and 129 (n_classes) outputs
W0 = tf.Variable(tf.random_normal([2 * n_neurons_in_h2, n_classes * embedding_dimension], mean=0, stddev=1/np.sqrt(n_features)), name="weightsOut")
b0 = tf.Variable(tf.random_normal([n_classes * embedding_dimension], mean=0, stddev=1/np.sqrt(n_features)), name="biasesOut")
#Activation function (sigmoid)
# 6400x129
a = tf.nn.sigmoid((tf.matmul(out_concate, W0)+b0), name="activationOutputLayer")

#reshape output of normal layer
# 6400 x 129 x 1
reshaped_emb = tf.reshape(a, [-1, n_classes, embedding_dimension])

# normalization before output
normalized_emb = tf.nn.l2_normalize(reshaped_emb, 2)

#%% Define Loss function

#Reshape output array to (n_features * frames_per_sample) x (number of dimensions) array
#returns a 64 x 12900 x 1 array
embedding_v = tf.reshape(normalized_emb, shape=[-1, frames_per_sample * n_features, embedding_dimension])

# get the Y(speaker indicator function)
#Reshape output array to (n_features * frames_per_sample) x (number of dimensions) array
#returns a 64 x 12900 x 2 array
Y_v = tf.reshape(Y, shape=[-1, frames_per_sample * n_features, 2])

# =============================================================================
# fast computation format of the embedding loss function
#returns a 64 x 1 x 1 array
loss_batch_X1 = tf.matmul(tf.transpose(embedding_v, [0, 2, 1]), embedding_v)
loss_batch_X2 = tf.nn.l2_loss(loss_batch_X1)

#returns a 64 x 1 x 2 array
loss_batch_XY1 = tf.matmul(tf.transpose(embedding_v, [0, 2, 1]), Y_v)
loss_batch_XY2 = 2 * tf.nn.l2_loss(loss_batch_XY1)

#returns a 64 x 2 x 2 array
loss_batch_Y1 = tf.matmul(tf.transpose(Y_v, [0, 2, 1]), Y_v)
loss_batch_Y2 = tf.nn.l2_loss(loss_batch_Y1)

loss_batch = ((loss_batch_X2 - loss_batch_XY2) + loss_batch_Y2) / batch_size

# =============================================================================
# old loss function (mse)

#Y2_v1 is used for the accuracy measure
#Y2_v1 has shape (6400 x 129)
Y2_v1 = tf.reshape(tf.slice(Y, [0, 0, 0], [-1, -1, 1]), shape=[-1, n_features])
#loss = tf.losses.mean_squared_error(a, Y2_v1)

# =============================================================================

#%% Apply loss function to gradients

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Separate out minimize operation so that gradient clipping can be applied
gradients_and_vars = optimizer.compute_gradients(loss_batch)
train_step = optimizer.apply_gradients(gradients_and_vars)


#%% Run model

#for timing how long model takes to train
import time

#folder for writer location
#Make sure there is a folder called WriterOutput held in the current directory
foldername = "WriterOutput"

#Location for model checkpoint
#Save to current directory
model_checkpoint = "./model.chkpt"


# Create a saver so we can save and load the model as we train it
tf_saver = tf.train.Saver()

initial = tf.global_variables_initializer()

#create session
with tf.Session() as sess:
    sess.run(initial)
    writer = tf.summary.FileWriter(foldername)
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()
    #Array for storing validation loss
    validation_loss_results= []
    #Arrays for storing training loss
    loss_results = []
    
    
    for epoch in range(training_epochs):
        
        # Start timer
        start = time.time()
        
        #create array to hold intermediate results for accuracy and loss
        intermediate_accuracy = []
        intermediate_loss = []
         
        #loop through each batch in the dataset
        for i in range(0, total_batches):
            
            # Get data
            data = data_generator.gen_batch(batch_size);
            # Reshape training data
            # concatenate training samples together to get (64, 100, 129) array
            tr_features = np.concatenate([np.reshape(item['Sample'], [1, frames_per_sample, n_features]) for item in data])
            
            # concatenate labels together to get (64, 100, 129) array
            Y_labels = (np.concatenate([np.asarray(item['Target']) for item in data])).astype('int')
            
            _, loss1  = sess.run([train_step, loss_batch], feed_dict={X: tr_features, Y:Y_labels})
            #Add loss to intermediate array
            intermediate_loss.append(loss1)
            
        #Save model after every 50 epochs
        if epoch % 50 == 0:
            save_path = tf_saver.save(sess, model_checkpoint)
            
        #Do validation after every 10 epochs
        if epoch % 2 == 0:
            
            validation_intermediate_loss = []
            
            #loop through each batch in the dataset
            for i in range(0, validation_batches):
                
                # Get data
                validation_data = validation_data_generator.gen_batch(batch_size);
                # Reshape training data
                # concatenate training samples together to get (64, 100, 129) array
                ts_features = np.concatenate([np.reshape(item['Sample'], [1, frames_per_sample, n_features]) for item in validation_data])
                
                # concatenate labels together to get (64, 100, 129) array
                ts_labels = (np.concatenate([np.asarray(item['Target']) for item in validation_data])).astype('int')
                
                validation_loss  = sess.run([loss_batch], feed_dict={X: ts_features, Y:ts_labels})
                #Add loss to intermediate array
                validation_intermediate_loss.append(validation_loss)
            
            #Append mean of loss  over batch to accuracy_loss results
            validation_loss_results.append([epoch, np.mean(intermediate_loss)])
            
        # finish timer 
        end = time.time()
        
        #Append mean of loss  over batch to accuracy_loss results
        loss_results.append([epoch ,np.mean(intermediate_loss), end - start])
        
        print("epoch %d, time elapsed: %f"%(epoch,(end - start)))
     
    #Save model after final epochs
    save_path = tf_saver.save(sess, model_checkpoint)

#Convert results to arrays
loss_results = np.asarray(loss_results)
validation_loss_results = np.asarray(validation_loss_results)            
            
print("Time elapsed (secs): %f"%(sum(loss_results[:,2])))

#Save results to csv
#Save training loss
np.savetxt("loss_results.csv", loss_results, delimiter=",")

#Save validation loss
np.savetxt("loss_validation_results.csv", validation_loss_results, delimiter=",")

del start, end


#%% remove variables
    
del i, loss1, intermediate_accuracy, intermediate_loss, epoch
del foldername, model_checkpoint
del save_path, learning_rate, batch_size, n_classes, n_features, n_neurons_in_h1, n_neurons_in_h2
del Y_labels, data, tr_features
del total_batches, total_number_of_datapoints
del states1, states2
del merged_summary

#%% Plot training accuracy and loss
import matplotlib.pyplot as plt

fig = plt.figure() 

#sub plot 1 - Loss
ax1 = plt.subplot(211)
plt.plot(range(training_epochs), loss_results[:,1], linewidth=2.0)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('training loss')


#sub plot 2 - Loss
ax2 = plt.subplot(212)
plt.plot(range(len(validation_loss_results)), validation_loss_results[:,1], linewidth=2.0)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('validation loss')

plt.tight_layout()
fig.savefig("deepclusteringloss.png", bbox_inches="tight")
plt.show()
plt.close(fig)  

#%%

del training_epochs, loss_results, validation_loss_results

#%% Test network with unseen data

#%% Create test data

#import DataGenerator class
import sys
sys.path.append('../Datagenerator')
from datagenerator import DataGenerator

wavfile1 = "../../TIMIT_WAV/Train/DR1/FCJF0/SI1027.WAV" 
wavfile2 = "../../TIMIT_WAV/Train/DR1/MDPK0/SI552.WAV" 

sampling_rate = 8000
frame_size = 256
maxFFTSize = 129

vad_threshold= 40


# create instance of class
data_generator = DataGenerator() 

#Get spectrogram of two wav files combined
testdata = data_generator.CreateTrainingDataSpectrogram(wavfile1, wavfile2, sampling_rate, frame_size, maxFFTSize, vad_threshold)


#%% Get embeddings for test signal from model

# Get only active test features
ts_features = testdata['Sample'][testdata['VAD']]

#Get number of features variable
n_features = np.shape(ts_features)[1]
  
#Previously saved model 
model_checkpoint = "./model.chkpt"

#Create array for storing IBM
embeddings = []

#create saver to save session
saver = tf.train.Saver()
   
with tf.Session() as sess:
    
    #restore session from checkpoint
    saver.restore(sess, model_checkpoint)
    
        
    #get length of spectrogram for mixture signal
    len_spec = ts_features.shape[0]
    k = 0
    
    #loop through spectrograms creating chunks of (frames_per_sample) time periods 
    while(k < len_spec):
        
        #if have come to the end of the spectrogram, need to pad the rest of the spectrogram with 0s to get a 129x209 array
        if (k + frames_per_sample > len_spec):
            # Get remaining data
            x = ts_features[k:k + frames_per_sample, :]
            
            # get shape of current ts_features
            current_len_spec = x.shape[0]
            
            # Pad ts_features so that it is the full size
            x = np.pad(x, ((0, (frames_per_sample-current_len_spec)), (0, 0)), 'constant', constant_values=(0))
            
            # reshape spectrogram for neural network
            x = np.reshape(x, [1, frames_per_sample, n_features])

        else:
            # Get data
            x = np.reshape(ts_features[k:k + frames_per_sample, :], [1, frames_per_sample, n_features])
            
        # get inferred ibm using trained model
        embeddings_batch = sess.run([embedding_v], feed_dict={X: x})
        
        # append ibm from batch to previous ibms
        # if have come to the end of the spectrogram, only append relevant points and not padded points
        if (k + frames_per_sample > len_spec):
            embeddings.append(embeddings_batch[0][:,0:(current_len_spec * n_features),:])
        else:
            embeddings.append(embeddings_batch[0])
        
        #increment k to look at next n time points
        k = k + frames_per_sample


# Convert list to array   
# this returns a 1 dimensional aarray of 1x21801       
embeddings_list = np.squeeze(np.concatenate([item for item in embeddings], axis=1), axis=0)

del k, current_len_spec

#%% Visualize the embeddings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#ax = Axes3D(fig)

#Plot embeddings
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embeddings_list[:,0], embeddings_list[:,1], embeddings_list[:,2], c = 'b', marker='o', alpha=0.5)


ax.set_xlabel('Embedding 1')
ax.set_ylabel('Embedding 2')
ax.set_zlabel('Embedding 3')
ax.set_title('Embeddings')


fig.savefig("deepclustering_embeddings.png", bbox_inches="tight")
plt.show()
plt.close(fig)  



#%% Apply k-means to test signal

from sklearn.cluster import KMeans

# Apply k-means to test signal
kmean = KMeans(n_clusters=2, random_state=0).fit(embeddings_list)

#%% Plot k-means

#Get 2 series from embeddings
x1 = embeddings_list[kmean.labels_==0,:]
x2 = embeddings_list[kmean.labels_==1,:]

#Plot embeddings
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1[:,0], x1[:,1], x1[:,2], c='blue', marker='o', alpha=0.5, label = 'cluster 1')
ax.scatter(x2[:,0], x2[:,1], x2[:,2], c='red', marker='o', alpha=0.5, label = 'cluster 2')

ax.set_xlabel('Embedding 1')
ax.set_ylabel('Embedding 2')
ax.set_zlabel('Embedding 3')
ax.set_title('Embeddings')
#ax.legend(['Cluster 1', 'Cluster 2'])
#ax.legend(['label1', 'label2'], numpoints = 2)
ax.legend()

fig.savefig("deepclustering_separation.png", bbox_inches="tight")
plt.show()
plt.close(fig) 


del x1, x2

#%% create ibm mask

ibm = np.transpose(kmean.labels_.reshape(-1,n_features).astype('int'))

#Add extra points to IBM where VAD is 0
#Get index of all time points that have 0 activity
vad = np.where(testdata['VAD'] == False)[0]
#take index away from index points
vad1 = np.subtract(vad, np.asarray(range(vad.shape[0])))
#Use insert to insert extra columns into ibm where VAD is 0
ibm = np.insert(ibm, vad1, 0, axis=1)

#%% Show original mixture spectrogram and mask

# Plot Spectrogram
fig = plt.figure() 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))


x = np.asarray(list(range(ibm.shape[0]))).astype(np.float) / ibm.shape[1]

ax1.pcolormesh(testdata['tmixed'], testdata['fmixed'], (np.abs(testdata['ZmixedSeries'])).transpose())
ax1.set(title='Mixture signal', xlabel='Time [sec]', ylabel='Frequency [Hz]')

ax2.imshow(ibm, cmap='Greys', interpolation='none', extent=[0,(209/129),129,0], aspect="auto")
ax2.set(title='IBM', xlabel='Time [sec]')

plt.tight_layout()
fig.savefig("deep_clustering_ibm.png", bbox_inches="tight")
plt.show()
plt.close(fig)

#%% apply ibm to signal and convert back into time domain

#Convert spectrogram from log to normal
split_spectrogram = np.power(10, testdata['Sample'])

#Apply ibm to spectrogram
split_spectrogram1 = np.multiply(split_spectrogram, np.transpose(ibm))

# create instance of class
data_generator = DataGenerator() 

#Get recovered mixture from istft routine
wav_recovered = data_generator.istft(split_spectrogram1, sampling_rate, frame_size)
   
#Amplify wav file
wav_recovered = np.float32(wav_recovered * 5)

#%% plot signal 

fig = plt.figure() 

#sub plot 1 - Original signal
ax1 = plt.subplot(211)
plt.plot(testdata['MixtureSignal'])
plt.xlabel('time')
plt.title('Mixture signal')

#sub plot 2 - Separated signal
ax2 = plt.subplot(212, sharex=ax1)
plt.plot(wav_recovered)
plt.xlabel('time')
plt.title('Separated signal')

#Save figure
plt.tight_layout()
fig.savefig("deep_clustering_recoveredwav.png", bbox_inches="tight")
plt.show()
plt.close(fig)

#%% Calculate signal to distortion ratio
import math

Signal1 = testdata['Signal1']
Signal2 = testdata['Signal2']
MixtureSignal = testdata['MixtureSignal']

top1 = (1 / Signal1.shape[0]) * np.sum(np.power(Signal1, 2))
top2 = (1 / Signal2.shape[0]) * np.sum(np.power(Signal2, 2))
bottom = (1 / MixtureSignal.shape[0]) * np.sum(np.power(MixtureSignal, 2))

SND1 = 10 * math.log10(top1/bottom)
SND2 = 10 * math.log10(top2/bottom)

objects = ('Signal 1', 'Signal 2')
y_pos = np.arange(len(objects))

fig = plt.figure() 

plt.bar(y_pos, [SND1, SND2], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Signal to distortion')
plt.title('Signal performance')
 
plt.show()

plt.close(fig)



#%% write final sound file to disk and play
import winsound
import librosa

#Save combined series to wav file
librosa.output.write_wav('separated_signal_rnn.wav', wav_recovered, sr=sampling_rate)

#play sound recovered
winsound.PlaySound('separated_signal_rnn.wav', winsound.SND_FILENAME|winsound.SND_ASYNC)









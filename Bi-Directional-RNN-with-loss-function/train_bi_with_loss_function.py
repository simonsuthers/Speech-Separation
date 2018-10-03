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
pkl_list = ['../Data/train' + str(i) + '.pkl' for i in range(1, 2)]

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
embedding_dimension = 1

#%% Placeholders for inputs, outputs, and Voice Activity detection matrix

#X = tf.placeholder(tf.float32, [batch_size, frames_per_sample, n_features], name="features")
X = tf.placeholder(tf.float32, shape=[None, frames_per_sample, n_features], name="features")

#Y = tf.placeholder(tf.float32, shape=[(batch_size * frames_per_sample), n_classes], name="labels")
Y = tf.placeholder(tf.float32, shape=[(batch_size * frames_per_sample), n_classes, 2], name="labels")

VAD = tf.placeholder(tf.float32, shape=[None, frames_per_sample, n_features], name="VAD")

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
W0 = tf.Variable(tf.random_normal([2 * n_neurons_in_h2, n_classes], mean=0, stddev=1/np.sqrt(n_features)), name="weightsOut")
b0 = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=1/np.sqrt(n_features)), name="biasesOut")
#Activation function (sigmoid)
# 6400x129
a = tf.nn.sigmoid((tf.matmul(out_concate, W0)+b0), name="activationOutputLayer")

#reshape output of normal layer
# 6400 x 129 x 1
reshaped_emb = tf.reshape(a, [-1, n_classes, embedding_dimension])

# normalization before output
normalized_emb = tf.nn.l2_normalize(reshaped_emb, 2)

#%% Define Loss function

# array with all outputs in a row
# 825,600 (6400 * 129) x 1
embedding_rs = tf.reshape(normalized_emb, shape=[-1, embedding_dimension])

# array with all VAD outputs in a row
# 825,600 (6400 * 129) x 1 array
VAD_rs = tf.reshape(VAD, shape=[-1])

# get only the embeddings with active VAD
# Multiply embeddings by VAD array
# returns # 825,600 (6400 * 129) x 1 array
embedding_rsv = tf.transpose(tf.multiply(tf.transpose(embedding_rs), VAD_rs))
#Reshape output array to (n_features * frames_per_sample) x (number of dimensions) array
#returns a 64 x 12900 x 1 array
embedding_v = tf.reshape(embedding_rsv, [-1, frames_per_sample * n_features, embedding_dimension])
 
# get the Y(speaker indicator function) with active VAD
# reshape Y to 825,600 (64 * 129 * 100) x 2 (number of speakers)
Y_rs = tf.reshape(Y, shape=[-1, 2])
# Multiply Y by VAD array to get only Y with active voice
Y_rsv = tf.transpose(tf.multiply(tf.transpose(Y_rs), VAD_rs))
#Reshape output array to (n_features * frames_per_sample) x (number of dimensions) array
#returns a 64 x 12900 x 2 array
Y_v = tf.reshape(Y_rsv, shape=[-1, frames_per_sample * n_features, 2])

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

#%% Accuracy calculation

#Get prediction from output
correct_prediction = tf.equal(tf.round(a), Y2_v1)
#Accuracy determination
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")


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

# Start timer
start = time.time()


initial = tf.global_variables_initializer()

#create session
with tf.Session() as sess:
    sess.run(initial)
    writer = tf.summary.FileWriter(foldername)
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()
    accuracy_results = []
    loss_results = []
    
    for epoch in range(training_epochs):
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
            
            # concatenate VAD arrays together to get (64, 100, 129) array
            VAD_reshaped = (np.concatenate([np.reshape(item['VAD'], [1, frames_per_sample, n_features]) for item in data])).astype('int')
            
            # concatenate labels together to get (64, 100, 129) array
            #Y_labels = (np.concatenate([np.reshape(np.asarray(item['Target'])[:,:,0], [1, frames_per_sample, n_features]) for item in data], axis=0)).astype(int)
            #Y_labels = (np.concatenate([np.asarray(item['Target'])[:,:,0] for item in data], axis=0)).astype(int)
            
            Y_labels = (np.concatenate([np.asarray(item['Target']) for item in data])).astype('int')
            
            _, loss1, accuracy1, loss_batch_Y1_1, loss_batch_Y2_1 = sess.run([train_step, loss_batch, accuracy, loss_batch_Y1, loss_batch_Y2], feed_dict={X: tr_features, Y:Y_labels, VAD: VAD_reshaped})
            #Add loss and accuracy to intermediate array
            intermediate_loss.append(loss1)
            intermediate_accuracy.append(accuracy1)
            
        #Append mean of loss and accuracy over batch to accuracy_loss results
        accuracy_results.append(np.mean(intermediate_accuracy))
        loss_results.append(np.mean(intermediate_loss))

        #Get prediction of validation data after each epoch
            
        #y_pred = sess.run(tf.argmax(a,1), feed_dict={X: ts_features})
        #y_true = sess.run(tf.argmax(ts_labels, 1))
        #summary, acc = sess.run([merged_summary, accuracy], feed_dict={X: ts_features, Y: ts_labels})
        
        #writer.add_summary(summary, epoch)
        print("epoch", epoch)
    save_path = tf_saver.save(sess, model_checkpoint)
    
# finish timer and display how long it took to build model in seconds
end = time.time()
print("Time elapsed (secs): %f"%(end - start))

del start, end
    
#%% remove variables
    
del i, loss1, accuracy1, intermediate_accuracy, intermediate_loss, epoch
del foldername, model_checkpoint
del save_path, learning_rate, batch_size, n_classes, n_features, n_neurons_in_h1, n_neurons_in_h2
del Y_labels, data, tr_features
del total_batches, total_number_of_datapoints
del states1, states2
del merged_summary

#%% Plot training accuracy and loss
import matplotlib.pyplot as plt

fig = plt.figure() 

#sub plot 1 - Accuracy
ax1 = plt.subplot(211)
plt.plot(range(training_epochs), accuracy_results, linewidth=2.0)
plt.xlabel('epoch')
plt.ylabel('training accuracy')
plt.title('accuracy')

#sub plot 2 - Loss
ax2 = plt.subplot(212, sharex=ax1)
plt.plot(range(training_epochs), loss_results, linewidth=2.0)
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.title('loss')

plt.tight_layout()
fig.savefig("bidilossaccuracy.png", bbox_inches="tight")
plt.show()
plt.close(fig)  

#%%

del training_epochs, accuracy_results, loss_results

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

#%% Create ibm from model

#Get number of features variable
n_features = np.shape(testdata['Sample'])[1]
  
#Previously saved model 
model_checkpoint = "./model.chkpt"

#Create array for storing IBM
ibm = []

#create saver to save session
saver = tf.train.Saver()
   
with tf.Session() as sess:
    
    #restore session from checkpoint
    saver.restore(sess, model_checkpoint)
        
    #get length of spectrogram for mixture signal
    len_spec = testdata['Sample'].shape[0]
    k = 0
    
    #loop through spectrograms creating chunks of (frames_per_sample) time periods 
    while(k < len_spec):
        
        #if have come to the end of the spectrogram, need to pad the rest of the spectrogram with 0s to get a 129x209 array
        if (k + frames_per_sample > len_spec):
            # Get remaining data
            ts_features = testdata['Sample'][k:k + frames_per_sample, :]
            
            # get shape of current ts_features
            current_len_spec = ts_features.shape[0]
            
            # Pad ts_features so that it is the full size
            ts_features = np.pad(ts_features, ((0, (frames_per_sample-current_len_spec)), (0, 0)), 'constant', constant_values=(0))
            
            # reshape spectrogram for neural network
            ts_features = np.reshape(ts_features, [1, frames_per_sample, n_features])

        else:
            # Get data
            ts_features = np.reshape(testdata['Sample'][k:k + frames_per_sample, :], [1, frames_per_sample, n_features])
                
        # get inferred ibm using trained model
        ibm_batch = sess.run([a], feed_dict={X: ts_features})
        
        # append ibm from batch to previous ibms
        # if have come to the end of the spectrogram, only append relevant points and not padded points
        if (k + frames_per_sample > len_spec):
            ibm.append(ibm_batch[0][0:current_len_spec,:])
        else:
            ibm.append(ibm_batch[0])
        
        #increment k to look at next n time points
        k = k + frames_per_sample


 
#Convert list to array       
ibm = np.concatenate([item for item in ibm], axis=0)

#Round each point to 0 or 1
ibm = (np.round(ibm, 0)).astype(int)
        
del k, n_features, len_spec, current_len_spec

#%% Show original mixture spectrogram and mask

# Plot Spectrogram
fig = plt.figure() 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))


x = np.asarray(list(range(ibm.shape[0]))).astype(np.float) / ibm.shape[1]

ax1.pcolormesh(testdata['tmixed'], testdata['fmixed'], (np.abs(testdata['ZmixedSeries'])).transpose())
ax1.set(title='Mixture signal', xlabel='Time [sec]', ylabel='Frequency [Hz]')

ax2.imshow(ibm.transpose(), cmap='Greys', interpolation='none', extent=[0,(209/129),129,0], aspect="auto")
ax2.set(title='IBM', xlabel='Time [sec]')

plt.tight_layout()
fig.savefig("bidilossibm.png", bbox_inches="tight")
plt.show()
plt.close(fig)

#%% apply ibm to signal and convert back into time domain

#Convert spectrogram from log to normal
split_spectrogram = np.power(10, testdata['Sample'])

#Apply ibm to spectrogram
split_spectrogram1 = np.multiply(split_spectrogram, ibm)

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
fig.savefig("bidilossrecoveredwav.png", bbox_inches="tight")
plt.show()
plt.close(fig)



#%% write final sound file to disk and play
import winsound
import librosa

#Save combined series to wav file
librosa.output.write_wav('separated_signal_rnn.wav', wav_recovered, sr=sampling_rate)

#play sound recovered
winsound.PlaySound('separated_signal_rnn.wav', winsound.SND_FILENAME|winsound.SND_ASYNC)









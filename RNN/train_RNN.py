'''
Script to train the model
'''


#https://becominghuman.ai/creating-your-own-neural-network-using-tensorflow-fa8ca7cc4d0e

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


#%% Placeholders for inputs and outputs

#X = tf.placeholder(tf.float32, [batch_size, frames_per_sample, n_features], name="features")
X = tf.placeholder(tf.float32, [None, frames_per_sample, n_features], name="features")

Y = tf.placeholder(tf.float32, [(batch_size * frames_per_sample), n_classes], name="labels")

#seq_length = tf.placeholder(tf.int32, [None])

#%% Layer 1

#Create lstm cell with n hidden neurons
with tf.variable_scope('BLSTM1') as scope: 
    lstm_fw_cell1 = tf.contrib.rnn.BasicLSTMCell(n_neurons_in_h1)
      
    #Create first layer
    #Produces a 300x100 output layer
    outputs1, states1 = tf.nn.dynamic_rnn(cell=lstm_fw_cell1, inputs=X, dtype=tf.float32)

#%% Layer 2
with tf.variable_scope('BLSTM2') as scope: 
    lstm_fw_cell2 = tf.contrib.rnn.BasicLSTMCell(n_neurons_in_h2)
      
    #Create first layer
    #Produces a 300x100 output layer
    outputs2, states2 = tf.nn.dynamic_rnn(cell=lstm_fw_cell2, inputs=outputs1, dtype=tf.float32)
    #Produces a 6400x300 output layer
    out_concate = tf.reshape(outputs2, [-1, n_neurons_in_h2])

#%% ouput layer

W0 = tf.Variable(tf.random_normal([n_neurons_in_h2, n_classes], mean=0, stddev=1/np.sqrt(n_features)), name="weightsOut")
b0 = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=1/np.sqrt(n_features)), name="biasesOut")
#Activation function (sigmoid)
a = tf.nn.sigmoid((tf.matmul(out_concate, W0)+b0), name="activationOutputLayer")

#%% Cost function

#Cost function
#MSE
#Y (64, 100, 129)
loss = tf.losses.mean_squared_error(a, Y)

#Optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

#%% Accuracy calculation

#Get prediction from output
correct_prediction = tf.equal(tf.round(a), Y)
#Accuracy determination
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

#correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#%% Run model


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
            # concatenate labels together to get (64, 100, 129) array
            #Y_labels = (np.concatenate([np.reshape(np.asarray(item['Target'])[:,:,0], [1, frames_per_sample, n_features]) for item in data], axis=0)).astype(int)
            #
            Y_labels = (np.concatenate([np.asarray(item['Target'])[:,:,0] for item in data], axis=0)).astype(int)
            
            _, loss1, accuracy1 = sess.run([train_step, loss, accuracy], feed_dict={X: tr_features, Y:Y_labels})
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
fig.savefig("rnnaccuracy.png", bbox_inches="tight")
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

# Get only active test features
ts_features = testdata['Sample'][testdata['VAD']]

#Get number of features variable
n_features = np.shape(ts_features)[1]
  
#Previously saved model 
model_checkpoint = "./model.chkpt"

#Create array for storing IBM
ibm_list = []

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
        ibm_batch = sess.run([a], feed_dict={X: x})
        
        # append ibm from batch to previous ibms
        # if have come to the end of the spectrogram, only append relevant points and not padded points
        if (k + frames_per_sample > len_spec):
            ibm_list.append(ibm_batch[0][0:current_len_spec,:])
        else:
            ibm_list.append(ibm_batch[0])
        
        #increment k to look at next n time points
        k = k + frames_per_sample


 
#Convert list to array       
ibm_array = np.concatenate([item for item in ibm_list], axis=0)

#Round each point to 0 or 1
ibm_array = (np.round(ibm_array, 0)).astype(int)

#Add extra points to IBM where VAD is 0
#Get index of all time points that have 0 activity
vad = np.where(testdata['VAD'] == False)[0]
#take index away from index points
vad1 = np.subtract(vad, np.asarray(range(vad.shape[0])))
#Use insert to insert extra columns into ibm where VAD is 0
ibm = np.insert(ibm_array, vad1, 0, axis=0)
        
del k, n_features, len_spec, current_len_spec, ibm_array

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
fig.savefig("rnnibm.png", bbox_inches="tight")
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
fig.savefig("rnnrecoveredwav.png", bbox_inches="tight")
plt.show()
plt.close(fig)



#%% write final sound file to disk and play
import winsound
import librosa

#Save combined series to wav file
librosa.output.write_wav('separated_signal_rnn.wav', wav_recovered, sr=sampling_rate)

#play sound recovered
winsound.PlaySound('separated_signal_rnn.wav', winsound.SND_FILENAME|winsound.SND_ASYNC)









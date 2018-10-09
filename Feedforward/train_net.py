'''
Script to train the model
'''

#Sources:
#https://becominghuman.ai/creating-your-own-neural-network-using-tensorflow-fa8ca7cc4d0e

import tensorflow as tf
import numpy as np

#%% Set console working directory
    
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

del wd

#%%

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

# Get sample data
data = data_generator.gen_batch(batch_size);
# Reshape training data
# concatenate training samples together to get 101900 x 129 array
tr_features = np.concatenate([item['Sample'] for item in data], axis=0)
# concatenate labels together to get 101900 x 129 array
Y_labels = (np.concatenate([np.asarray(item['Target'])[:,:,0] for item in data], axis=0)).astype(int)

#Same as neff
n_features = np.shape(tr_features)[1]
#Same as neff
n_classes = np.shape(Y_labels)[1]

del data, tr_features, Y_labels

#%% Reset default graph

tf.reset_default_graph()

#%% Set training parameters

training_epochs = 50
n_neurons_in_h1 = 300
n_neurons_in_h2 = 300
learning_rate = 0.01

#%% Placeholders for inputs and outputs

X = tf.placeholder(tf.float32, [None, n_features], name="features")
Y = tf.placeholder(tf.float32, [None, n_classes], name="labels")

#%% Layer 1

W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1/np.sqrt(n_features)), name="weights1")
b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1], mean=0, stddev=1/np.sqrt(n_features)), name="biases1")
# Activation function
y1 = tf.nn.tanh((tf.matmul(X, W1)+b1), name="activationlayer1")

#%% Layer 2

W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2], mean=0, stddev=1/np.sqrt(n_features)), name="weights2")
b2 = tf.Variable(tf.random_normal([n_neurons_in_h2],mean=0, stddev=1/np.sqrt(n_features)), name="biases2")
#Activation function
y2 = tf.nn.tanh((tf.matmul(y1,W2)+b2), name="activationlayer2")

#%% ouput layer

W0 = tf.Variable(tf.random_normal([n_neurons_in_h2, n_classes], mean=0, stddev=1/np.sqrt(n_features)), name="weightsOut")
b0 = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=1/np.sqrt(n_features)), name="biasesOut")
#Activation function (tanh)
a = tf.nn.sigmoid((tf.matmul(y2, W0)+b0), name="activationOutputLayer")

#%% Cost function

#Cost function
#cross_entropy
#loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(a), reduction_indices=[1]))
#MSE
loss = tf.losses.mean_squared_error(Y, a)

#Optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

#%% Accuracy calculation

#Get prediction from output
correct_prediction = tf.equal(tf.round(a), Y)
#Accuracy determination
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")


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
            # concatenate training samples together to get 101900 x 129 array
            tr_features = np.concatenate([item['Sample'] for item in data], axis=0)
            # concatenate labels together to get 101900 x 129 array
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
del save_path, learning_rate, n_classes, n_features, n_neurons_in_h1, n_neurons_in_h2
del Y_labels, data, tr_features
del total_batches, merged_summary, batch_size

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
fig.savefig("feedforwardaccuracy.png", bbox_inches="tight")
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

#%% As a test - convert spectrogram back into time domain
import matplotlib.pyplot as plt

#Get spectrogram from testing routine
testspectrogram1 = np.power(10, testdata['Sample'])

#Get recovered mixture from istft routine
wav_recovered = data_generator.istft(testspectrogram1, sampling_rate, frame_size)
   
#Amplify wav file
wav_recovered = wav_recovered * 5

#See wav file
fig = plt.figure() 

#sub plot 1 - Original signal
ax1 = plt.subplot(211)
plt.plot(testdata['MixtureSignal'])
plt.xlabel('time')
plt.title('original signal')

#sub plot 2 - Recovered signal
ax2 = plt.subplot(212, sharex=ax1)
plt.plot(wav_recovered)
plt.xlabel('time')
plt.title('recovered signal')

plt.show() 
plt.close(fig)

#%% remove unused variables

del testspectrogram1, wav_recovered

#%% Create ibm from model

#Data for model
ts_features = testdata['Sample'][testdata['VAD']]
  
#Previously saved model 
model_checkpoint = "./model.chkpt"


#create saver to save session
saver = tf.train.Saver()
   
with tf.Session() as sess:
        
    #restore session from checkpoint
    saver.restore(sess, model_checkpoint)
    
    # get inferred ibm using trained model
    ibm_batch = sess.run([a], feed_dict={X: ts_features})
        
 
#Convert list to array       
ibm_array = np.concatenate([item for item in ibm_batch], axis=0)

#Round each point to 0 or 1
ibm_array = (np.round(ibm_array, 0)).astype(int)

#Add extra points to IBM where VAD is 0
vad = np.where(testdata['VAD'] == False)[0]
vad1 = np.subtract(vad, np.asarray(range(vad.shape[0])))
ibm = np.insert(ibm_array, vad1, 0, axis=0)


#del ibm_array, vad, vad1

#%% Show mask

fig = plt.figure() 
plt.imshow(ibm.transpose(), cmap='Greys', interpolation='none')

plt.tight_layout()
fig.savefig("feedforwardibm2.png", bbox_inches="tight")
plt.show()
plt.close(fig)

#%% Show VAD mask

vad = np.tile(testdata['VAD'], (129, 1))

fig = plt.figure() 

plt.imshow(vad, cmap='Greys', interpolation='none', extent=[0,(209/129),129,0], aspect="auto")
plt.xlabel('Time [sec]')
plt.title('VAD matrix')

plt.tight_layout()
fig.savefig("feedforwardVAD.png", bbox_inches="tight")
plt.show()
plt.close(fig)

del vad

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
fig.savefig("feedforwardibm.png", bbox_inches="tight")
plt.show()
plt.close(fig)


#%% apply ibm to signal and convert back into time domain

#Convert spectrogram from log to normal
split_spectrogram1 = np.power(10, ts_features)

#Add extra points to IBM whre VAD is 0
vad = np.where(testdata['VAD'] == False)[0]
vad1 = np.subtract(vad, np.asarray(range(vad.shape[0])))
split_spectrogram = np.insert(split_spectrogram1, vad1, 0, axis=0)

#Apply ibm to spectrogram
split_spectrogram1 = np.multiply(split_spectrogram, ibm)

# create instance of class
data_generator = DataGenerator() 

#Get recovered mixture from istft routine
wav_recovered = data_generator.istft(split_spectrogram1, sampling_rate, frame_size)
   
#Amplify wav file
wav_recovered = np.float32(wav_recovered * 5)

del split_spectrogram1, vad, vad1

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
fig.savefig("feedforwardrecoveredwav.png", bbox_inches="tight")
plt.show()
plt.close(fig)

#%% write final sound file to disk and play
import winsound
import librosa

#Save combined series to wav file
librosa.output.write_wav('separated_signal.wav', wav_recovered, sr=sampling_rate)

#play sound recovered
winsound.PlaySound('separated_signal.wav', winsound.SND_FILENAME|winsound.SND_ASYNC)

#del filepath








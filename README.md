# Speech Separation using Neural Networks and Tensorflow

## Introduction

The files experiment Speech separation using various neural network structures. The experiments feed in a dataset of sound files with containing 2 clean voices and attempt to build a network to separate out the 2 voices.
So far, 2 networks have been built:
* Feed forward network
* RNN network

The scripts were created using the Spyder IDE of anaconda. Before executing each script, set the console directory to the directory of the script.

## Data Generator

Within the **DataGenerator** folder are two Python scripts that create the dataset.
It is assumed that a top-level folder exists called TIMIT_WAV that contains the TIMIT dataset. The top-level folder should look something like this:

![alt text](https://github.com/simonsuthers/Speech-Separation/blob/master/Pictures/FolderStructure.png?raw=true "Folder structure")

### datagenerator.py
The **datagenerator.py** script contains a class to create the data set. The dataset is saved as several pickle files. Each pickle file contains 
The pickle files are saved to a top level folder called **Data**.
### datagenerator2.py
The **datagenerator2.py** takes the data from a given number of pickle files and feeds data into tensorflow session in batches. 

## Feed Forward network
### train_net.py

The feedforward folder contains a python script called **train_net.py** that trains a feedforward network. The network contains 2 hidden layers of 300 neurons and an output layer of 129 neurons (one for each frequency bin in the spectrogram). 
The output layer uses a sigmoid activation function. A mean squared error loss function is used.
After 50 epochs, the network struggles to find any pattern in the data. The accuracy after 50 epochs is still close to 50%.

![Alt text](Speech-Separation/Feedforward/feedforwardaccuracy.png?raw=true "Feedforward accuracy")

A test signal containing mixture of 2 voices was fed into the network and the following IBM was produced:

![Alt text](Speech-Separation/Feedforward/feedforwardibm.png?raw=true "Feedforward IBM")

After applying the IBM, the original sound wave looks (and sounds) the same as the original sound wave, implying that a feed forward network is not a good model for speech separation.

![Alt text](Speech-Separation/Feedforward/feedforwardrecoveredwav.png?raw=true "Feedforward recovered sound wave")

## RNN network
### train_RNN.py

The RNN folder contains a python script called **train_rnn.py**. This scripts trains a 2 layer RNN using LSTM cells containing 300 neurons. A final feedforward layer with 129 neurons using a sigmoid activation function produces an IBM. A mean squared error loss function was used.

The network uses the same **datagenerator.py** class to create the data. The spectrograms are split into chunks of 100 time frequency bins which are fed into the RNN. The remainder data in a spectrogram after the nearest value of 100 is not used for training. 
Like the feed forward network, the network struggles to separate the two sound sources. Accuracy on the training set after 50 epochs is still almost 50%.

![alt text](https://github.com/simonsuthers/Speech-Separation/tree/master/Speech-Separation/SpeechSegregation/rnnaccuracy.png "RNN accuracy")

As with the feed forward network, a test signal containing mixture of 2 voices was fed into the network and the following IBM was produced:

![alt text](https://github.com/simonsuthers/Speech-Separation/tree/master/Speech-Separation/SpeechSegregation/rnnibm.png "RNN ibm")

As with the feed forward network, after applying the IBM, the original sound wave looks (and sounds) the same as the original sound wave, implying that a feed forward network is not a good model for speech separation.

![alt text](https://github.com/simonsuthers/Speech-Separation/tree/master/Speech-Separation/SpeechSegregation/ rnnrecoveredwav.png "RNN recovered sound wave")


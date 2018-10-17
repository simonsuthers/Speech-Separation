# Speech Separation using Neural Networks and Tensorflow

## Introduction

The files experiment Speech separation using various neural network structures. The experiments feed in a dataset of sound files with containing 2 clean voices and attempt to build a network to separate out the 2 voices.
So far, 2 networks have been built:
* Feed forward network
* RNN network

The scripts were created using the Spyder IDE of anaconda. Before executing each script, set the console directory to the directory of the script.

## Source

J. R. Hershey, Z. Chen, J. Le Roux and S. Watanabe, "Deep clustering: Discriminative embeddings for segmentation and separation," 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Shanghai, 2016, pp. 31-35

## Data Generator

Within the **DataGenerator** folder are two Python scripts that create the dataset.
It is assumed that a top-level folder exists called TIMIT_WAV that contains the TIMIT dataset. The top-level folder should look something like this:

![Alt text](Pictures/FolderStructure.png?raw=true "Folder structure")

### datagenerator.py
The **datagenerator.py** script contains a class to create the data set. The dataset is saved as several pickle files. Each pickle file contains 
The pickle files are saved to a top level folder called **Data**.
### datagenerator2.py
The **datagenerator2.py** takes the data from a given number of pickle files and feeds data into tensorflow session in batches. 

## Feed Forward network
### train_net.py

The feedforward folder contains a python script called **train_net.py** that trains a feedforward network. The network contains 2 hidden layers of 300 neurons and an output layer of 129 neurons (one for each frequency bin in the spectrogram). 
The output layer uses a sigmoid activation function. A mean squared error loss function is used on a known IBM. The following schematic represents the flow of code:

![Alt text](Pictures/Feed_forward.png?raw=true "Feedforward flow")

After 50 epochs, the network struggles to find any pattern in the data. The accuracy after 50 epochs is still close to 50%.

![Alt text](Feedforward/feedforwardaccuracy.png?raw=true "Feedforward accuracy")

A test signal containing mixture of 2 voices was fed into the network and the following IBM was produced:

![Alt text](Feedforward/feedforwardibm.png?raw=true "Feedforward IBM")

After applying the IBM, the original sound wave looks (and sounds) the same as the original sound wave, implying that a feed forward network is not a good model for speech separation.

![Alt text](Feedforward/feedforwardrecoveredwav.png?raw=true "Feedforward recovered sound wave")

## RNN network
### train_RNN.py

The RNN folder contains a python script called **train_rnn.py**. This scripts trains a 2 layer RNN using LSTM cells containing 300 neurons. A final feedforward layer with 129 neurons using a sigmoid activation function produces an IBM. A mean squared error loss function was used against a known IBM. The flow is shown in the following schematic:

![Alt text](Pictures/RNN.png?raw=true "RNN flow")

The network uses the same **datagenerator.py** class to create the data. The spectrograms are split into chunks of 100 time frequency bins which are fed into the RNN. The remainder data in a spectrogram after the nearest value of 100 is not used for training. 
Like the feed forward network, the network struggles to separate the two sound sources. Accuracy on the training set after 50 epochs is still almost 50%.

![Alt text](RNN/rnnaccuracy.png?raw=true "RNN accuracy")

As with the feed forward network, a test signal containing mixture of 2 voices was fed into the network and the following IBM was produced:

![Alt text](RNN/rnnibm.png?raw=true "RNN IBM")

As with the feed forward network, after applying the IBM, the original sound wave looks (and sounds) the same as the original sound wave, implying that a RNN network is not a good model for speech separation.

![Alt text](RNN/rnnrecoveredwav.png?raw=true "RNN recovered sound wave")

## Bi-directional RNN network
### train_bi_directional_RNN.py

The Bi-Directional-RNN folder contains a python script called **train_bi_directional_RNN.py**. This scripts trains a 2 layer bi-directional RNN using LSTM cells containing 300 neurons. A final feedforward layer with 129 neurons using a sigmoid activation function produces an IBM. A mean squared error loss function was used against a known IBM. The flow is shown in the following schematic:

![Alt text](Pictures/Bi-directional-RNN.png?raw=true "Bi-directional RNN flow")

As with the one-directional RNN, the network uses the same **datagenerator.py** class to create the data. The spectrograms are split into chunks of 100 time frequency bins which are fed into the RNN. The remainder data in a spectrogram after the nearest value of 100 is not used for training. 
Accuracy on the training set after 50 epochs is still only 50%.

![Alt text](Bi-Directional-RNN/bidirnnaccuracy.png?raw=true "Bi-directional RNN accuracy")

The same test signal containing mixture of 2 voices was fed into the network and the following IBM was produced:

![Alt text](Bi-Directional-RNN/bidirnnibm.png?raw=true "Bi-directional RNN IBM")

As with the other networks, after applying the IBM, the original sound wave looks (and sounds) the same as the original sound wave, implying that a bi-directional RNN network on its own is not a good model for speech separation.

![Alt text](Bi-Directional-RNN/bidirnnrecoveredwav.png?raw=true " Bi-directional RNN recovered sound wave")


## Bi-directional RNN network with deep clustering loss function
### train_bi_with_loss_function.py

The Bi-Directional-RNN-with-loss-function folder contains a python script called **train_bi_with_loss_function.py**. This scripts trains the same 2 layer bi-directional RNN as before. This time, the loss function from deep clustering was implemented. The flow is shown in the following schematic:

![Alt text](Pictures/Bi-directional-RNN-with-loss-function.png?raw=true "Bi-directional RNN with loss function flow")

Accuracy on the training set after 50 epochs was erratic. However, the purpose of the loss function is to move neurons in the final layer apart.

![Alt text](Bi-Directional-RNN-with-loss-function/bidilossaccuracy.png?raw=true "Bi-directional RNN with loss function accuracy")

The same test signal containing mixture of 2 voices was fed into the network and the following IBM was produced:

![Alt text](Bi-Directional-RNN-with-loss-function/bidilossibm.png?raw=true "Bi-directional RNN with loss function IBM")

As with the other networks, after applying the IBM, the original sound wave looks (and sounds) the same as the original sound wave, implying that a bi-directional RNN network on its own is not a good model for speech separation.

![Alt text](Bi-Directional-RNN-with-loss-function/bidilossrecoveredwav.png?raw=true " Bi-directional RNN with loss function recovered sound wave")

## Full deep clustering model with k-means clustering
### train_deep_clustering.py

The full deep-clustering model in simplemented in the **Deep-clustering** folder within the python script called **train_deep_clustering.py**. The programmatic flow is shown in the following schematic:

![Alt text](Pictures/deep_clustering.png?raw=true "Deep clustering flow")

The bi-directional LSTM model created before creates embeddings. Test signals are then fed into these embeddings. An example of the embeddings from a test signal is shown below:

![Alt text](Deep-clustering/deepclustering_embeddings.png?raw=true "Deep clustering embeddings")

K-means clustering is then applied to the embeddings to assign each embedding a speaker:

![Alt text](Deep-clustering/deepclustering_separation.png?raw=true "Deep clustering separation")

The loss function is designed to move embeddings from different sources further apaert and embeddings from the same source closer together:

![Alt text](Deep-clustering/deepclusteringloss.png?raw=true "Deep clustering loss function")

The same test signal containing mixture of 2 voices as before was fed into the network. Clustering was performed on the results and the following IBM was produced:

![Alt text](Deep-clustering/deep_clustering_ibm.png?raw=true "Deep clustering IBM")

Below is the output of the binary mask. If enough data is fed into the network, some separation is audible (honest!):

![Alt text](Deep-clustering/deep_clustering_recoveredwav.png?raw=true "Deep clustering recovered sound wave")

## Full deep clustering model with mean-shift clustering

![Alt text](Deep-clustering-with-mean-shift-clustering/deepclustering_embeddings.png?raw=true "Deep clustering embeddings")

![Alt text](Deep-clustering-with-mean-shift-clustering/deepclustering_separation.png?raw=true "Deep clustering separation")

![Alt text](Pictures/snr_comparison.png?raw=true "SNR results")






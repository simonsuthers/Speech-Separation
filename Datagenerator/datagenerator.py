#%% Script to create pickle files for each folder of wav files


#%%
import numpy as np
import librosa
import pickle
import os
from scipy import signal
#import math

#%% Class to combine wav files and save combinations of wav files as pickle files

class DataGenerator(object):
    
    #Constructor
    def __init__(self):
        '''class to combine wav files, create spectrogram of wav files and save
        combination of files as pickle files'''
        #self.ind = 0  # index of current reading position
        #self.samples = []
        #self.epoch = 0
        

    #Create method to apply a stft to a sound array
    def stft(self, wavfile, sampling_rate, frame_size, overlapFac=0.75):
        """ short time fourier transform of audio signal """
        
        #Set window size
        overlap = int(overlapFac*frame_size)
        #Get stft
        f1, t1, Zsamples1 = signal.stft(wavfile, fs=sampling_rate, window='hann', nperseg = frame_size, return_onesided=True, noverlap=overlap, axis=1)
        #f1, t1, Zsamples1 = signal.stft(wavfile, fs=sampling_rate, window='hann', nperseg = frame_size, return_onesided=True, noverlap=overlap)
    
        return Zsamples1, f1, t1
    
    #Create method to apply inverse stft to a spectorgram created using the stft method
    def istft(self, spectrogram, sampling_rate, frame_size, overlapFac=0.75):
        """ short time fourier transform of audio signal """
        
        #Set window size
        overlap = int(overlapFac*frame_size)
        #Get istft
        t1, samplesrec = signal.istft(spectrogram, fs=sampling_rate, window='hann', nperseg = frame_size, input_onesided = True, noverlap=overlap, time_axis=-2, freq_axis=-1)
        #t1, samplesrec = signal.istft(spectrogram, fs=sampling_rate, window='hann', nperseg = frame_size, input_onesided = True, noverlap=overlap)
    
        return samplesrec
    
    #Function that returns a list of all wav files in all sub folders within a main folder
    #Function returns an array with eachrow representing a subfoler. Each row contians a lsit of wav files in that folder.
    def GetListOfFiles(self, data_dir):
        '''Gets a list of all wav files within a directory'''
        
        #Get folders contained within data_dir folder
        speakers_dir = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
        
        #count of folders within data_dir folder
        n_speaker = len(speakers_dir)
        speaker_file = {}
    
        # get the files in each speakers dir
        # for each folder in data_dir folder, add wav file name and wav file path to speaker_file dict
        for i in range(n_speaker):
            #get list of files in folder
            wav_dir_i = [os.path.join(speakers_dir[i], file) for file in os.listdir(speakers_dir[i])]
            
            
            if i not in speaker_file:
                speaker_file[i] = []
    
            #for each wav file in folder, add to speaker_file dict
            for j in wav_dir_i:
                if j.endswith(".WAV"):
                    speaker_file[i].append(j)
                    
        return n_speaker, speaker_file
    
    #Function to join 2 wav files together and create single dictionary of {ID, wavfile1, wavfile2, samples, VAD, Target}
    def CreateTrainingDataSpectrogram(self, wavfile1, wavfile2, sampling_rate, frame_size, maxFFTSize, vad_threshold):
        #load first speaker
        speech_1, _ = librosa.core.load(wavfile1, sr=sampling_rate)
        #load second speaker
        speech_2, _ = librosa.core.load(wavfile2, sr=sampling_rate)
        
        # mix 2 speech signals together
        # Reduce length of signals so that they are a multiple of 128
        length = min(len(speech_1), len(speech_2))
        #length = int(math.floor(length / 128.0)) * 128
        speech_1 = speech_1[:length]
        speech_2 = speech_2[:length]
        speech_mix = speech_1 + speech_2
        
        # compute stft spectrum for 1st speaker
        speech_1_spec, _, _ = self.stft(speech_1, sampling_rate, frame_size)
        speech_1_spec = np.abs(speech_1_spec[:, :maxFFTSize])
        
        # compute stft spectrum for 2nd speaker
        speech_2_spec, _, _ = self.stft(speech_2, sampling_rate, frame_size)
        speech_2_spec = np.abs(speech_2_spec[:, :maxFFTSize])
        
        # compute log stft spectrum for mixture of both speaker
        speech_mix_spec0, f1, t1 = self.stft(speech_mix, sampling_rate, frame_size)
        speech_mix_spec1 = np.abs(speech_mix_spec0[:, :maxFFTSize])
        # Get additional frequency spectrum for mixture signal
        speech_mix_spec = np.log10(speech_mix_spec1)
        #Convert speech_mix_spec to float16 to save on memory
        speech_mix_spec = speech_mix_spec.astype('float16')
               
        # VAD is voice activity detection. If magnitude is greater than threshold then a voice is active.
        speech_VAD = (speech_mix_spec1.sum(axis=1) > 0.005).astype(int)
        #Convert VAD to boolean
        speech_VAD = speech_VAD.astype(bool)
        
        # Create Ideal Binary Mask
        # 2 IBMs are created. One for the first signal and one for the second signal
        Y = np.array([speech_1_spec > speech_2_spec, speech_1_spec < speech_2_spec]).astype(bool)
        #Transpose IBM around so that it is 2 columns of n frequency points for each time point
        Y1 = np.transpose(Y, [1, 2, 0])
        
        sample_dict = {'Sample': speech_mix_spec, 'VAD': speech_VAD, 'Target': Y1, 'Wavfiles': [wavfile1, wavfile2], 'MixtureSignal': speech_mix, 'ZmixedSeries': speech_mix_spec0, 'fmixed': f1, 'tmixed': t1}

        return sample_dict
    
    
    #Function to create pickle files from a list of folders in a file path
    def CreatePickleFiles(self, filepath, data_dir, sampling_rate, maxFFTSize, frame_size, vad_threshold, frames_per_sample):
        '''Init the training data using the wav files'''
        
        #Get list of directories to create speech sample from
        n_speaker, speaker_file = self.GetListOfFiles(data_dir)
        
        speaker_file_match = {}
        
        # generate match dict which randomly matches 2 wav files together
        # The resulting dict has a key of a wav file and a value of a randomly picked second wav file
        # match each file in each folder with another random file
       
        #loop through each top level folder
        for i in range(n_speaker):
            #for each wav file in foler 
            for j in speaker_file[i]:
                #randomly choose another folder
                k = np.random.randint(n_speaker)
                # make sure it is not the same fiolder
                while(i == k):
                    k = np.random.randint(n_speaker)
                # randomly choose another wav file in the randomly chosen folder
                l = np.random.randint(len(speaker_file[k]))
                # assign random wav file to current wav file
                speaker_file_match[j] = speaker_file[k][l]
    
        #Array for holding all samples in
        samples = [] 
        
        #id varaible to hold id of mixture of wav files
        id = 1
        
        # for each file pair, generate their mixture and reference samples
        for i in speaker_file_match:
            j = speaker_file_match[i]
            
            #Create sample dictionary
            sample = self.CreateTrainingDataSpectrogram(i, j, sampling_rate, frame_size, maxFFTSize, vad_threshold)  
            
            #reduce spectrogram to only include bins with activity greater than threshold
            train = sample['Sample'][sample['VAD']]
            Y = sample['Target'][sample['VAD']]
            
            #get length of spectrogram for mixture signal
            len_spec = train.shape[0]
            k = 0
            vad_start = 0
    
            #loop through spectrograms creating chunks of (frames_per_sample) time periods 
            while(k + frames_per_sample < len_spec):
                #Get first chunk of data from spectrogram of 3 signals
                #Chunk splits the spectrogram into a series of n (FRAMES_PER_SAMPLE) points
                speech_mix_spec = train[k:k + frames_per_sample, :]
                Y1 = Y[k:k + frames_per_sample, :]
                
                #Get VAD values for all points to k
                vad_end = np.where(sample['VAD'] == True)[0][k + frames_per_sample]
                speech_VAD = sample['VAD'][vad_start:vad_end]
                vad_start = vad_end
                
                #Create feed_dict for neural network
                sample_dict = {'Sample': speech_mix_spec, 'VAD': speech_VAD, 'Target': Y1, 'Wavfiles': sample['Wavfiles'], 'Id':id }

                #Add sample dictionary to list of samples
                samples.append(sample_dict)
                
                #increment k to look at next n time points
                k = k + frames_per_sample
                
            #Increment id to next number
            id = id + 1
        
        # dump the generated sample list
        pickle.dump(samples, open(filepath, 'wb'))
        
        
    #function to loop through all training folders and create pickle file for all folders
    def CreatePickleForAllFolders(self, parentfolder, sampling_rate, maxFFTSize, frame_size, vad_threshold, endfolder, frames_per_sample, filefolder):
        
        training_folders = [parentfolder + '\\DR' + str(i) for i in range(1, (endfolder + 1))]

        #loop through list of folders
        for i in training_folders:
            #Get folder name
            data_dir = i
            #Create name of file to save data to
            filename = "%strain%s.pkl"%(filefolder, i[-1:])

            #Save data to pickle file
            self.CreatePickleFiles(filename, data_dir, sampling_rate, maxFFTSize, frame_size, vad_threshold, frames_per_sample)
            
            #Print message to confirm creation
            print("file created: %s"%filename)

#%% Unit test
#Set console working directory
            
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

del wd

#%% Test stft and isft function

#Set file
file = "../../TIMIT_WAV/Train/DR1/FCJF0/SI1027.WAV"

# create instance of class
data_generator = DataGenerator() 

#Set sampling rate and frame size
sampling_rate = 8000
frame_size = 256


#Load wav file
wavfile1, _ = librosa.load(file, sr=sampling_rate, dtype=float)
 
#Create stft using scipy
spectrogram1, _, _ = data_generator.stft(wavfile1, sampling_rate, frame_size)

#Convert spectrogram back to time domain and check result matches original
wavfile2 = data_generator.istft(spectrogram1, sampling_rate, frame_size)

#Convert spectrogram back to time domain using real component only
wavfile3 = data_generator.istft(np.abs(spectrogram1), sampling_rate, frame_size)

#%% Delete variables 
del file, wavfile1, wavfile2, wavfile3, spectrogram1, sampling_rate, frame_size, data_generator

#%% Test GetListOfFiles function
# Should return a list of files in an array    
   
#Set directory to get list of files from
data_dir = "../../TIMIT_WAV/Train/DR1" 

# create instance of class
data_generator = DataGenerator() 

n_speaker, speaker_file = data_generator.GetListOfFiles(data_dir)

#%% Delete variables
del data_dir, n_speaker, speaker_file, data_generator

#%% Test create CreateTrainingDataSpectrogram function
#Create Spectrogram

  
frame_size = 256
maxFFTSize = 129
sampling_rate = 8000
vad_threshold = 8000
#Set wav files to combine
wavfile1 = "../../TIMIT_WAV/Train/DR1/FCJF0/SI1027.WAV" 
wavfile2 = "../../TIMIT_WAV/Train/DR1/MDPK0/SI552.WAV" 

# create instance of class
data_generator = DataGenerator() 
#Routine
sample = data_generator.CreateTrainingDataSpectrogram(wavfile1, wavfile2, sampling_rate, frame_size, maxFFTSize, vad_threshold)   

#Test converting Spectrogram back to time domain 
#Get spectrogram from testing routine
spectrogram = np.power(10, sample['Sample'])
#Get original mixture from testing routine
wav_original = sample['MixtureSignal']
#Get recovered mixture from istft routine
wav_recovered = data_generator.istft(spectrogram, sampling_rate, frame_size)   
wav_recovered = np.float32(wav_recovered * 5)


#%% Test create CreateTrainingDataSpectrogram function
#Show chart comparing recovered signal
import matplotlib.pyplot as plt

plt.figure(1)

#sub plot 1 - Original signal
ax1 = plt.subplot(211)
plt.plot(wav_original)
plt.xlabel('time')
plt.title('original signal')

#sub plot 2 - Recovered signal
ax2 = plt.subplot(212, sharex=ax1)
plt.plot(wav_recovered)
plt.xlabel('time')
plt.title('recovered signal')

plt.show()   

#%% Save and play 
import winsound

#Save combined series to wav file
#Wav files will be saved to the current working directory
librosa.output.write_wav('original_mix.wav', wav_original, sr=sampling_rate)
librosa.output.write_wav('recovered_mix.wav', wav_recovered, sr=sampling_rate)

#play sound recovered
winsound.PlaySound('recovered_mix.wav', winsound.SND_FILENAME|winsound.SND_ASYNC)

#del filepath

#%% remove testing variables

del data_generator
del frame_size, maxFFTSize, sampling_rate, vad_threshold, wavfile1, wavfile2, sample, spectrogram, wav_original, wav_recovered

#%% Test CreatePickleFiles function

#Directory of wav files
data_dir = "../../TIMIT_WAV/Train/DR8" 

#Location to save pickle files to
#Save pickle files to current working directory
filename = "../Data/test.pkl"

#Sampling rate
sampling_rate = 8000
#Maximum number of FFT points (NEFF)
maxFFTSize = 129
#Size of FFT frame
frame_size = 256
#Voice activity threshold (THRESHOLD)
#If TF bins are smaller than THRESHOLD then will be considered inactive
vad_threshold = 40
#Number of frames per smaple for batches (this will be the nimber of rows that are fed into each batch of the rnn)
frames_per_sample = 100

# create instance of class
data_generator = DataGenerator() 

# Run CreatePickleFiles function
#data_generator.CreatePickleFiles(filename, data_dir, sampling_rate, maxFFTSize, frame_size, vad_threshold, frames_per_sample)

#%% Clear variables

del filename, sampling_rate, maxFFTSize, frame_size, vad_threshold, data_dir, frames_per_sample

#%% Create training set
#Use the folder DR1 to DR7 to create the training set

#Sampling rate
sampling_rate = 8000
#Maximum number of FFT points (NEFF)
maxFFTSize = 129
#Size of FFT frame
frame_size = 256

#Voice activity threshold (THRESHOLD)
#If TF bins are smaller than THRESHOLD then will be considered inactive
vad_threshold = 40
#Number of frames per sample for batches (this will be the nimber of rows that are fed into each batch of the rnn)
frames_per_sample = 100
 
#Set folder containing folders of wav files from TIMIT data set 
parentfolder = "../../TIMIT_WAV/Train" 

#Set folder to save pickle files
#Folder is current working directory
filefolder = "../Data/"

#number of folders to loop through
endfolder = 7

# create instance of class
data_generator = DataGenerator() 

# Run routine to create pickle files
#data_generator.CreatePickleForAllFolders(parentfolder, sampling_rate, maxFFTSize, frame_size, vad_threshold, endfolder, frames_per_sample, filefolder)

  
#%% Remove variables

del parentfolder, sampling_rate, maxFFTSize, frame_size, vad_threshold, data_generator, frames_per_sample, endfolder, filefolder


   

    
    
     

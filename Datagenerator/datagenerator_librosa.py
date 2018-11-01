#%% Script to create pickle files for each folder of wav files


#%%
import numpy as np
import librosa
import pickle
import os
import matplotlib.pyplot as plt
import time
import librosa.display

#%% Class to combine wav files and save combinations of wav files as pickle files

class DataGenerator(object):
    
    #Constructor
    def __init__(self):
        '''class to combine wav files, create spectrogram of wav files and save
        combination of files as pickle files'''
        #set global mean and std (obtained from Get-mean-and-std.py script)
        self.mean = -1.623956
        self.std = 0.790590
        

    #Create method to apply a stft to a sound array
    def stft(self, wavfile, frame_size):
        """ short time fourier transform of audio signal """
        # Get stft
        stft = librosa.stft(y=wavfile, n_fft=frame_size, window='hann')
        
        # Get magnitude and phase
        mag, phase = librosa.magphase(stft)
        
        # transpose outputs for neural network
        stft = np.transpose(stft)
        mag = np.transpose(mag)
        phase = np.transpose(np.angle(phase))
    
        return stft, mag, phase
    
    #Create method to apply inverse stft to a spectorgram created using the stft method
    def istft(self, stft_matrix):
        """ short time fourier transform of audio signal """
        
        stft_matrix = np.transpose(stft_matrix)
        #Get istft
        istft = librosa.istft(stft_matrix=stft_matrix, window='hann')
    
        return istft
    
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
    
    #Function to return wav files
    def GetSeperateSignals(self, wavfile1, wavfile2, sampling_rate):
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
        
        return speech_1, speech_2, speech_mix
        
    # function that takes a magnitude array and phas array and returns a stft array
    def stft_from_mag_and_phase(self, radii, angle_radians):
        return radii * np.exp(1j*angle_radians)   
    
    #Function to join 2 wav files together and create single dictionary of {ID, wavfile1, wavfile2, samples, VAD, Target}
    def CreateTrainingDataSpectrogram(self, wavfile1, wavfile2, sampling_rate, frame_size, maxFFTSize, vad_threshold):

        #Get separate and mixed speech signals
        speech_1, speech_2, speech_mix = self.GetSeperateSignals(wavfile1, wavfile2, sampling_rate)
        
        # compute stft spectrum for 1st speaker
        speech_1_spec, speech_1_mag, speech_1_phase = self.stft(speech_1, frame_size)
        speech_1_mag = speech_1_mag[:, :maxFFTSize]
        speech_1_phase = speech_1_phase[:, :maxFFTSize]
        
        # compute stft spectrum for 2nd speaker
        speech_2_spec, speech_2_mag, speech_2_phase = self.stft(speech_2, frame_size)
        speech_2_mag = speech_2_mag[:, :maxFFTSize]
        speech_2_phase = speech_2_phase[:, :maxFFTSize]
        
        # compute log stft spectrum for mixture of both speaker
        speech_mix_spec0, speech_mix_mag0, speech_mix_phase0 = self.stft(speech_mix, frame_size)
        speech_mix_mag0 = speech_mix_mag0[:, :maxFFTSize]
        speech_mix_phase0 = speech_mix_phase0[:, :maxFFTSize]
        # Get additional frequency spectrum for mixture signal
        speech_mix_mag_log = np.log10(speech_mix_mag0)
        # Standardise spectrogram by minus mean and divide by standard deviation
        # Global mean and standard deviation are worked out in Get-mean-and-std.py script
        speech_mix_mag_std = (speech_mix_mag_log - self.mean) / self.std
        #Convert speech_mix_spec to float16 to save on memory
        speech_mix_mag_log = speech_mix_mag_log.astype('float16')
        speech_mix_mag_std = speech_mix_mag_std.astype('float16')
         
        # VAD is voice activity detection. If magnitude is greater than threshold then a voice is active.
        # Get maximum magnitude of mixture signal
        max_mag = np.max(speech_mix_mag0)
        speech_VAD = (speech_mix_mag0.sum(axis=1) > (max_mag * maxFFTSize * vad_threshold)).astype(int)
        #Convert VAD to boolean
        speech_VAD = speech_VAD.astype(bool)
        
        # Create Ideal Binary Mask
        # 2 IBMs are created. One for the first signal and one for the second signal
        IBM = np.array([speech_1_mag > speech_2_mag, speech_1_mag < speech_2_mag]).astype(bool)
        #Transpose IBM around so that it is 2 columns of n frequency points for each time point
        IBM = np.transpose(IBM, [1, 2, 0])
        
        # Create Ideal Ratio Mask
        # 2 IRMs are created. One for the first signal and one for the second signal
        SNR1 = np.log10(np.divide(speech_1_mag, speech_2_mag))
        SNR2 = np.log10(np.divide(speech_2_mag, speech_1_mag))
        IRM = np.array([(np.power(10, SNR1) / (np.power(10, SNR1) + 1)), (np.power(10, SNR2) / (np.power(10, SNR2) + 1))]).astype('float16')
        #Transpose IBM around so that it is 2 columns of n frequency points for each time point
        IRM = np.transpose(IRM, [1, 2, 0])
        
        sample_dict = {'SampleStd': speech_mix_mag_std, 'SampleLog': speech_mix_mag_log, 'SamplePhaseRaw': speech_mix_phase0, 'SampleMagRaw': speech_mix_mag0, 'VAD': speech_VAD, 'IBM': IBM, 'IRM': IRM,'Wavfiles': [wavfile1, wavfile2], 'MixtureSignal': speech_mix, 'Signal1': speech_1, 'Signal2': speech_2, 'Speech1Magnitude': speech_1_mag, 'Speech2Magnitude': speech_2_mag }

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
            trainStd = sample['SampleStd'][sample['VAD']]
            trainLog = sample['SampleLog'][sample['VAD']]
            IBM = sample['IBM'][sample['VAD']]
            IRM = sample['IRM'][sample['VAD']]
            
            #get length of spectrogram for mixture signal
            len_spec = trainStd.shape[0]
            k = 0
            vad_start = 0
    
            #loop through spectrograms creating chunks of (frames_per_sample) time periods 
            while(k + frames_per_sample < len_spec):
                #Get first chunk of data from spectrogram of 3 signals
                #Chunk splits the spectrogram into a series of n (FRAMES_PER_SAMPLE) points
                speech_mix_spec_Std = trainStd[k:k + frames_per_sample, :]
                speech_mix_spec_Log = trainLog[k:k + frames_per_sample, :]
                IBM1 = IBM[k:k + frames_per_sample, :]
                IRM1 = IRM[k:k + frames_per_sample, :]
                
                #Get VAD values for all points to k
                vad_end = np.where(sample['VAD'] == True)[0][k + frames_per_sample]
                speech_VAD = sample['VAD'][vad_start:vad_end]
                vad_start = vad_end
                
                #Create feed_dict for neural network
                sample_dict = {'SampleStd': speech_mix_spec_Std, 'SampleLog': speech_mix_spec_Log, 'VAD': speech_VAD, 'IBM': IBM1, 'IRM': IRM1, 'Wavfiles': sample['Wavfiles'], 'Id':id }

                #Add sample dictionary to list of samples
                samples.append(sample_dict)
                
                #increment k to look at next n time points
                k = k + frames_per_sample
                
            #Increment id to next number
            id = id + 1
        
        # dump the generated sample list
        pickle.dump(samples, open(filepath, 'wb'))
        
        
    #function to loop through all training folders and create pickle file for all folders
    def CreatePickleForAllFolders(self, file, parentfolder, sampling_rate, maxFFTSize, frame_size, vad_threshold, endfolder, frames_per_sample, filefolder):
        
        training_folders = [parentfolder + '/DR' + str(i) for i in range(1, (endfolder + 1))]

        #loop through list of folders
        for i in training_folders:
            #Get folder name
            data_dir = i
            #Create name of file to save data to
            filename = "%s%s%s.pkl"%(filefolder, file, i[-1:])

            #Save data to pickle file
            self.CreatePickleFiles(filename, data_dir, sampling_rate, maxFFTSize, frame_size, vad_threshold, frames_per_sample)
            
            #Print message to confirm creation
            print("file created: %s"%filename)

#%% Unit tests
    
#%% Test stft and isft function

#Set file
file = "../../TIMIT_WAV/Train/DR1/FCJF0/SI1027.WAV"

# create instance of class
data_generator = DataGenerator() 

# Set sampling rate and frame size
sampling_rate = 8000
frame_size = 256


# Load wav file
wavfile1, _ = librosa.load(file, sr=sampling_rate, dtype=float)
 
# Create stft using scipy
stft, mag1, phase1 = data_generator.stft(wavfile1, frame_size)

# Convert spectrogram back to time domain and check result matches original
wavfile2 = data_generator.istft(stft)

# Convert spectrogram back to time domain using real component only
wavfile3 = data_generator.istft(mag1)

# Recreate stft from magnitude and phase
stft2 = data_generator.stft_from_mag_and_phase(mag1, phase1)

#print results
print("wavfile1 shape:")
print(wavfile1.shape)
print("wavfile2 shape:")
print(wavfile2.shape)
print("wavfile3 shape:")
print(wavfile3.shape)
print("stft shape:")
print(stft.shape)
print("stft2 shape:")
print(stft2.shape)


#%% Show spectrogram

plt.figure()

librosa.display.specshow(librosa.amplitude_to_db(np.transpose(mag1), ref=np.max), sr=sampling_rate, hop_length=frame_size, y_axis='log', x_axis='time')

plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

#%% Show chart of results

#Show chart of results
fig = plt.figure() 
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5,7))

#sub plot 1 - Original signal
ax1.plot(wavfile1)
ax1.set(title='Original signal', xlabel='time')

#sub plot 2 - Recovered signal
ax2.plot(wavfile2)
ax2.set(title='recovered signal', xlabel='time')

#sub plot 3 - Recovered signal (real component only)
ax3.plot(wavfile3)
ax3.set(title='recovered signal (real component only)', xlabel='time')

plt.tight_layout()
plt.show()
plt.close(fig) 

#%% Delete variables 
del file, wavfile1, wavfile2, wavfile3, stft, stft2, sampling_rate, mag1, phase1, \
frame_size, data_generator

#%% Test GetListOfFiles function
# Should return a list of files in an array    
   
# Set directory to get list of files from
data_dir = "../../TIMIT_WAV/Train/DR1" 

# create instance of class
data_generator = DataGenerator() 

n_speaker, speaker_file = data_generator.GetListOfFiles(data_dir)

print("speaker_file length : %d"%len(speaker_file))
print("n_speaker size : %d"%n_speaker)
print(speaker_file[0])

#%% Delete variables
del data_dir, n_speaker, speaker_file, data_generator

#%% Test create CreateTrainingDataSpectrogram function
# Create Spectrogram

frame_size = 256
maxFFTSize = 129
sampling_rate = 8000
vad_threshold = 0.001
#Set wav files to combine
wavfile1 = "../../TIMIT_WAV/Train/DR1/FCJF0/SI1027.WAV" 
wavfile2 = "../../TIMIT_WAV/Train/DR1/MDPK0/SI552.WAV" 

# create instance of class
data_generator = DataGenerator() 
# Get sample data for to wav files
sample = data_generator.CreateTrainingDataSpectrogram(wavfile1, wavfile2, sampling_rate, frame_size, maxFFTSize, vad_threshold)   

# Test converting Spectrogram back to time domain 
# Get spectrogram from testing routine
# Convert from z score
stft = np.power(10, (sample['SampleStd'] * data_generator.std) + data_generator.mean)

# Get phase from testing routine
phase = sample['SamplePhaseRaw']

# Get original mixture from testing routine
wav_original = sample['MixtureSignal']

# Get recovered mixture from istft routine
wav_recovered = data_generator.istft(stft)  

# Get recovered mixture from istft routine with phase 
stft2 = data_generator.stft_from_mag_and_phase(stft, phase)
wav_recovered_with_phase = data_generator.istft(stft2)  

print("wav_original length: %d"%wav_original.shape[0])
print("wav_recovered length: %d"%wav_recovered.shape[0])
print("wav_recovered_with_phase length: %d"%wav_recovered_with_phase.shape[0])

#%% Test create CreateTrainingDataSpectrogram function
# Show chart comparing recovered signal


plt.figure(1)

# sub plot 1 - Original signal
ax1 = plt.subplot(311)
plt.plot(wav_original)
plt.xlabel('time')
plt.title('original signal')

# sub plot 2 - Recovered signal
ax2 = plt.subplot(312, sharex=ax1)
plt.plot(wav_recovered)
plt.xlabel('time')
plt.title('recovered signal')

# sub plot 3 - Recovered signal with phase
ax3 = plt.subplot(313, sharex=ax1)
plt.plot(wav_recovered_with_phase)
plt.xlabel('time')
plt.title('recovered signal with phase')

plt.tight_layout()
plt.show()   

#%% Save sounds as wav files

# Save combined series to wav file
# Wav files will be saved to the current working directory
librosa.output.write_wav('original_mix.wav', wav_original, sr=sampling_rate)
librosa.output.write_wav('recovered_mix.wav', wav_recovered, sr=sampling_rate)

#%% remove testing variables

del data_generator
del frame_size, maxFFTSize, sampling_rate, vad_threshold, wavfile1, wavfile2, sample, \
 stft, stft2, wav_original, wav_recovered, wav_recovered_with_phase, phase

#%% Test CreatePickleFiles function

# Directory of wav files
data_dir = "../../TIMIT_WAV/Train/DR8" 

# Location to save pickle files to
# Save pickle files to current working directory
filename = "../Data/test.pkl"

# Sampling rate
sampling_rate = 8000
# Maximum number of FFT points (NEFF)
maxFFTSize = 129
# Size of FFT frame
frame_size = 256
# Voice activity threshold (THRESHOLD)
# If TF bins are smaller than THRESHOLD then will be considered inactive
vad_threshold = 0.001
# Number of frames per smaple for batches (this will be the nimber of rows that are fed into each batch of the rnn)
frames_per_sample = 100

# create instance of class
data_generator = DataGenerator() 

# Start timer
start = time.time()

# Run CreatePickleFiles function
#data_generator.CreatePickleFiles(filename, data_dir, sampling_rate, maxFFTSize, frame_size, vad_threshold, frames_per_sample)

# finish timer 
end = time.time()

print("Created test pickle file. Time elapsed: %f"%(end - start))

  
#%% Test training set

test = pickle.load(open("../Data/test.pkl", 'rb'))

print("test length: %d"%len(test))

#%% Clear variables

del filename, sampling_rate, maxFFTSize, frame_size, vad_threshold, data_dir, frames_per_sample, \
    start, end, test

#%% Create training set
# Use the folder DR1 to DR7 to create the training set

# Sampling rate
sampling_rate = 8000
# Maximum number of FFT points (NEFF)
maxFFTSize = 129
# Size of FFT frame
frame_size = 256

# Voice activity threshold (THRESHOLD)
# If TF bins are smaller than THRESHOLD then will be considered inactive
vad_threshold = 0.001
# Number of frames per sample for batches (this will be the nimber of rows that are fed into each batch of the rnn)
frames_per_sample = 100
 
# Set folder containing folders of wav files from TIMIT data set 
parentfolder = "../../TIMIT_WAV/Train" 

# Set folder to save pickle files
# Folder is current working directory
filefolder = "../Data/"

# number of folders to loop through
endfolder = 8

# create instance of class
data_generator = DataGenerator() 

# Start timer
start = time.time()

# Run routine to create pickle files
#data_generator.CreatePickleForAllFolders("train", parentfolder, sampling_rate, maxFFTSize, frame_size, vad_threshold, endfolder, frames_per_sample, filefolder)

# finish timer 
end = time.time()

print("Created training pickle files. Time elapsed: %f"%(end - start))
  
#%% Test training set

train1 = pickle.load(open("../Data/train1.pkl", 'rb'))
train2 = pickle.load(open("../Data/train2.pkl", 'rb'))
train3 = pickle.load(open("../Data/train3.pkl", 'rb'))

print("train1 length: %d"%len(train1))
print("train2 length: %d"%len(train2))
print("train3 length: %d"%len(train3))

#%% Remove variables

del parentfolder, sampling_rate, maxFFTSize, frame_size, vad_threshold, data_generator, \
    frames_per_sample, endfolder, filefolder, start, end, train1, train2, train3

#%% Create testing set
# Use the folder DR1 to DR7 to create the training set

# Sampling rate
sampling_rate = 8000
# Maximum number of FFT points (NEFF)
maxFFTSize = 129
# Size of FFT frame
frame_size = 256

# Voice activity threshold (THRESHOLD)
# If TF bins are smaller than THRESHOLD then will be considered inactive
vad_threshold = 0.001
# Number of frames per sample for batches (this will be the nimber of rows that are fed into each batch of the rnn)
frames_per_sample = 100
 
# Set folder containing folders of wav files from TIMIT data set 
parentfolder = "../../TIMIT_WAV/TEST" 

# Set folder to save pickle files
# Folder is current working directory
filefolder = "../Data/"

# number of folders to loop through
endfolder = 8

# create instance of class
data_generator = DataGenerator() 

# Run routine to create pickle files
#data_generator.CreatePickleForAllFolders("test", parentfolder, sampling_rate, maxFFTSize, frame_size, vad_threshold, endfolder, frames_per_sample, filefolder)

#%% Test testing set

test1 = pickle.load(open("../Data/test1.pkl", 'rb'))
test2 = pickle.load(open("../Data/test2.pkl", 'rb'))
test3 = pickle.load(open("../Data/test3.pkl", 'rb'))

print("test1 length: %d"%len(test1))
print("test2 length: %d"%len(test2))
print("test3 length: %d"%len(test3))

   

    
    
     

#%% Script to create pickle files for each folder of wav files


#%%
import numpy as np
import librosa
import pickle
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import random

#%% Class to combine wav files and save combinations of wav files as pickle files

class DataGenerator(object):
    
    #Constructor
    def __init__(self):
        '''class to combine wav files, create spectrogram of wav files and save
        combination of files as pickle files'''
        #set global mean and std (obtained from Get-mean-and-std.py script)
        self.mean = -3.7185277644387273
        self.std = 0.7873899776985277
        

    #Create method to apply a stft to a sound array
    def stft(self, wavfile, sampling_rate, frame_size, overlapFac=0.75):
        """ short time fourier transform of audio signal """
        
        #Set window size
        overlap = int(overlapFac*frame_size)
        #Get stft
        f1, t1, Zsamples1 = signal.stft(wavfile, fs=sampling_rate, window='hann', nperseg = frame_size, return_onesided=True, noverlap=overlap, axis=1)
    
        return Zsamples1, f1, t1
    
    #Create method to apply inverse stft to a spectorgram created using the stft method
    def istft(self, spectrogram, sampling_rate, frame_size, overlapFac=0.75):
        """ short time fourier transform of audio signal """
        
        #Set window size
        overlap = int(overlapFac*frame_size)
        #Get istft
        t1, samplesrec = signal.istft(spectrogram, fs=sampling_rate, window='hann', nperseg = frame_size, input_onesided = True, noverlap=overlap, time_axis=-2, freq_axis=-1)

        return samplesrec
    
    #Function that returns a list of all wav files in all sub folders within a main folder
    #Function returns an array with eachrow representing a subfoler. Each row contians a lsit of wav files in that folder.
    def GetListOfFiles(self, data_dir, gender="X", number_of_matching_files = 3):
        '''Joins all wav files in a directory with all wav files in another random directory'''
        
        # Get folders contained within data_dir folder
        if gender=="X":
            speakers_dir = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
        else:
            # Just get single gender folders contained within data_dir folder
            speakers_dir = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if i.startswith(gender)]
                
        # count of folders within data_dir folder
        n_folders = len(speakers_dir)
        speaker_file = []
        
        # get the files in each speakers dir
        # for each folder in data_dir folder, add wav file name and wav file path to speaker_file dict
        for i in range(n_folders):
            # get list of files in folder
            files_in_folder = [os.path.join(speakers_dir[i], f1) for f1 in os.listdir(speakers_dir[i]) if f1.endswith(".WAV")]
            
            # for each wav file in folder, add all other speaker files to file
            for file1 in files_in_folder:
                # randomly choose another folder
                list_of_folders2 = [x for x in range(n_folders) if x!= i]
                folder2 = random.choice(list_of_folders2)
                
                # get list of files in secondary folder
                files_in_folder2 = [os.path.join(speakers_dir[folder2], file) for file in os.listdir(speakers_dir[folder2]) if file.endswith(".WAV")]
                # randomly select 3 files in the folder
                files_in_folder2_2 = random.sample(files_in_folder2, number_of_matching_files)
                
                # add each file in folder to first file
                for file2 in files_in_folder2_2:
                    file_dict = {'file1': file1, 'file2': file2}
                    speaker_file.append(file_dict)
                    
        length_of_speaker_file = len(speaker_file)
                    
        return n_folders, length_of_speaker_file, speaker_file
    
   
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
        
        return speech_1, speech_2, speech_mix, (length / sampling_rate)
        

    # function that takes a magnitude array and phas array and returns a stft array
    def stft_from_mag_and_phase(self, radii, angle_radians):
        return radii * np.exp(1j*angle_radians)   
        
    
    #Function to join 2 wav files together and create single dictionary of {ID, wavfile1, wavfile2, samples, VAD, Target}
    def CreateTrainingDataSpectrogram(self, wavfile1, wavfile2, sampling_rate, frame_size, maxFFTSize, vad_threshold):
        
        #Get separate and mixed speech signals
        speech_1, speech_2, speech_mix, signal_length = self.GetSeperateSignals(wavfile1, wavfile2, sampling_rate)
        
        # compute stft spectrum for 1st speaker
        speech_1_spec, _, _ = self.stft(speech_1, sampling_rate, frame_size)
        speech_1_mag = np.abs(speech_1_spec[:, :maxFFTSize])
        speech_1_phase_raw = np.angle(speech_1_spec[:, :maxFFTSize])
        speech_1_phase_std = speech_1_phase_raw / np.pi
        
        # compute stft spectrum for 2nd speaker
        speech_2_spec, _, _ = self.stft(speech_2, sampling_rate, frame_size)
        speech_2_mag = np.abs(speech_2_spec[:, :maxFFTSize])
        speech_2_phase_raw = np.angle(speech_2_spec[:, :maxFFTSize])
        speech_2_phase_std = speech_2_phase_raw / np.pi
        
        # compute log stft spectrum for mixture of both speaker
        speech_mix_spec0, f1, t1 = self.stft(speech_mix, sampling_rate, frame_size)
        speech_mix_mag = np.abs(speech_mix_spec0[:, :maxFFTSize])
        speech_mix_phase_raw = np.angle(speech_mix_spec0[:, :maxFFTSize])
        # Get additional frequency spectrum for mixture signal
        speech_mix_mag_log = np.log10(speech_mix_mag)
        # Standardise spectrogram by minus mean and divide by standard deviation
        # Global mean and standard deviation are worked out in Get-mean-and-std.py script
        speech_mix_mag_std = (speech_mix_mag_log - self.mean) / self.std
        speech_mix_phase_std = speech_mix_phase_raw / np.pi
        #Convert speech_mix_spec to float16 to save on memory
        speech_mix_mag_log = speech_mix_mag_log.astype('float16')
        speech_mix_mag_std = speech_mix_mag_std.astype('float16')
               
        # VAD is voice activity detection. If magnitude is greater than threshold then a voice is active.
        speech_VAD = (speech_mix_mag.sum(axis=1) > 0.005).astype(int)
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
        
        #sample_dict = {'SampleStd': speech_mix_mag_std, 'SampleLog': speech_mix_mag_log, 'VAD': speech_VAD, 'IBM': IBM1, 'IRM': IRM1,'Wavfiles': [wavfile1, wavfile2], 'MixtureSignal': speech_mix, 'ZmixedSeries': speech_mix_spec0, 'fmixed': f1, 'tmixed': t1, 'Signal1': speech_1, 'Signal2': speech_2, 'Spectrogram1': speech_1_mag, 'Spectrogram2':speech_2_mag }
        sample_dict = {'SampleStd': speech_mix_mag_std, 'SampleLog': speech_mix_mag_log, 'SamplePhaseRaw': speech_mix_phase_raw, \
                       'SamplePhaseStd':speech_mix_phase_std, 'SampleMagRaw': speech_mix_mag, 'VAD': speech_VAD, 'IBM': IBM, 'IRM': IRM, \
                       'Wavfiles': [wavfile1, wavfile2], 'MixtureSignal': speech_mix, 'ZmixedSeries': speech_mix_spec0, \
                       'fmixed': f1, 'tmixed': t1, 'Signal1': speech_1, 'Signal2': speech_2, 'Speech1Magnitude': speech_1_mag, \
                       'Speech2Magnitude': speech_2_mag, 'Speech1PhaseRaw': speech_1_phase_raw, 'Speech2PhaseRaw': speech_2_phase_raw, \
                       'Speech1PhaseStd': speech_1_phase_std, 'Speech2PhaseStd': speech_2_phase_std, \
                       'SignalLength': signal_length}


        return sample_dict
    
    
    #Function to create pickle files from a list of folders in a file path
    def CreatePickleFiles(self, filepath, data_dir, sampling_rate, maxFFTSize, frame_size, vad_threshold, frames_per_sample, gender="X", number_of_matching_files = 3):
        '''Init the training data using the wav files'''
        
        # Get list of files to create speech sample from
        n_folders, length_of_speaker_file, speaker_file = self.GetListOfFiles(data_dir, gender, number_of_matching_files)
    
        # Array for holding all samples in
        samples = [] 
        
        # id varaible to hold id of mixture of wav files
        id = 1
        
        # Variable to hold total signal length
        total_signal_length = 0
        
        # for each file pair, generate their mixture and reference samples
        for file in speaker_file:
            i = file['file1']
            j = file['file2']
            
            #Create sample dictionary
            sample = self.CreateTrainingDataSpectrogram(i, j, sampling_rate, frame_size, maxFFTSize, vad_threshold)  
            
            #reduce spectrogram to only include bins with activity greater than threshold
            trainStd = sample['SampleStd'][sample['VAD']]
            trainLog = sample['SampleLog'][sample['VAD']]
            trainPhaseStd = sample['SamplePhaseStd'][sample['VAD']]
            IBM = sample['IBM'][sample['VAD']]
            IRM = sample['IRM'][sample['VAD']]
            Y_PhaseStd = sample['Speech1PhaseStd'][sample['VAD']]
            
            # Add signal length to total signal length
            total_signal_length += sample['SignalLength']
            
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
                speech_mix_spec_Phase = trainPhaseStd[k:k + frames_per_sample, :]
                IBM1 = IBM[k:k + frames_per_sample, :]
                IRM1 = IRM[k:k + frames_per_sample, :]
                signal1_Phase = Y_PhaseStd[k:k + frames_per_sample, :]
                
                #Get VAD values for all points to k
                vad_end = np.where(sample['VAD'] == True)[0][k + frames_per_sample]
                speech_VAD = sample['VAD'][vad_start:vad_end]
                vad_start = vad_end
                
                #Create feed_dict for neural network
                sample_dict = {'SampleStd': speech_mix_spec_Std, 'SampleLog': speech_mix_spec_Log, 'SamplePhaseStd': speech_mix_spec_Phase, 'Y_Phase': signal1_Phase, 'VAD': speech_VAD, 'IBM': IBM1, 'IRM': IRM1, 'Wavfiles': sample['Wavfiles'], 'Id':id }

                #Add sample dictionary to list of samples
                samples.append(sample_dict)
                
                #increment k to look at next n time points
                k = k + frames_per_sample
                
            #Increment id to next number
            id = id + 1
        
        # dump the generated sample list
        pickle.dump(samples, open(filepath, 'wb'))
        print("Total signal length (hours): %f"%(total_signal_length/3600))
        
        
    #function to loop through all training folders and create pickle file for all folders
    def CreatePickleForAllFolders(self, file, parentfolder, sampling_rate, maxFFTSize, frame_size, vad_threshold, endfolder, frames_per_sample, filefolder, gender="X", number_of_matching_files = 3):
        
        training_folders = [parentfolder + '/DR' + str(i) for i in range(1, (endfolder + 1))]

        #loop through list of folders
        for i in training_folders:
            #Get folder name
            data_dir = i
            #Create name of file to save data to
            filename = "%s%s%s.pkl"%(filefolder, file, i[-1:])

            #Save data to pickle file
            self.CreatePickleFiles(filename, data_dir, sampling_rate, maxFFTSize, frame_size, vad_threshold, frames_per_sample, gender, number_of_matching_files)
            
            #Print message to confirm creation
            print("file created: %s"%filename)

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
stft, f1, t1 = data_generator.stft(wavfile1, sampling_rate, frame_size)

#Convert spectrogram back to time domain and check result matches original
wavfile2 = data_generator.istft(stft, sampling_rate, frame_size)

#Convert spectrogram back to time domain using real component only
wavfile3 = data_generator.istft(np.abs(stft), sampling_rate, frame_size)

# print results
print("wavfile1 shape:")
print(wavfile1.shape)
print("wavfile2 shape:")
print(wavfile2.shape)
print("wavfile3 shape:")
print(wavfile3.shape)
print("stft shape:")
print(stft.shape)

#%% Show spectrogram

plt.figure()

plt.pcolormesh(t1, f1, (np.abs(stft).transpose()))
plt.title='Mixture signal'
plt.xlabel='Time [sec]'
plt.ylabel='Frequency [Hz]'

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

del file, wavfile1, wavfile2, wavfile3, stft, sampling_rate, \
frame_size, data_generator, f1, t1


#%% Test GetListOfFiles function
# Should return a list of files in an array    
   
#Set directory to get list of files from
data_dir = "../../TIMIT_WAV/Train/DR1" 

# create instance of class
data_generator = DataGenerator() 

n_folders, length_of_speaker_file, speaker_file = data_generator.GetListOfFiles(data_dir)

print("speaker_file length : %d"%len(speaker_file))
print("n_folders size : %d"%n_folders)
print(speaker_file[0])

#%% Delete variables
del data_dir, n_folders, length_of_speaker_file, speaker_file, data_generator

#%% Test create GetSeperateSignals function

sampling_rate = 8000
#Set wav files to combine
wavfile1 = "../../TIMIT_WAV/Train/DR1/FCJF0/SI1027.WAV" 
wavfile2 = "../../TIMIT_WAV/Train/DR1/MDPK0/SI552.WAV" 

# create instance of class
data_generator = DataGenerator() 
#Routine
speech_1, speech_2, speech_mix, signal_length = data_generator.GetSeperateSignals(wavfile1, wavfile2, sampling_rate)

print("signal length (secs): %f"%signal_length)

#%% Delete variables
del sampling_rate, wavfile1, wavfile2, speech_1, speech_2, speech_mix, signal_length, data_generator

#%% Test create CreateTrainingDataSpectrogram function
#Create Spectrogram

frame_size = 256
maxFFTSize = 129
sampling_rate = 8000
vad_threshold = 0.001
#Set wav files to combine
wavfile1 = "../../TIMIT_WAV/Train/DR1/FCJF0/SI1027.WAV" 
wavfile2 = "../../TIMIT_WAV/Train/DR1/MDPK0/SI552.WAV" 

# create instance of class
data_generator = DataGenerator() 
#Routine
sample = data_generator.CreateTrainingDataSpectrogram(wavfile1, wavfile2, sampling_rate, frame_size, maxFFTSize, vad_threshold)   

#Test converting Spectrogram back to time domain 
#Convert from z score
stft = np.power(10, (sample['SampleStd'] * data_generator.std) + data_generator.mean)

# Get phase from testing routine
phase = sample['SamplePhaseRaw']

#Get original mixture from testing routine
wav_original = sample['MixtureSignal']

#Get recovered mixture from istft routine
wav_recovered = data_generator.istft(stft, sampling_rate, frame_size) 

# Get recovered mixture from istft routine with phase 
stft2 = data_generator.stft_from_mag_and_phase(stft, phase)
wav_recovered_with_phase = data_generator.istft(stft2, sampling_rate, frame_size)    

print("wav_original length: %d"%wav_original.shape[0])
print("wav_recovered length: %d"%wav_recovered.shape[0])
print("wav_recovered_with_phase length: %d"%wav_recovered_with_phase.shape[0])


#%% Test create CreateTrainingDataSpectrogram function
# Show chart comparing recovered signal

plt.figure(figsize=(6,6))

# Show chart of results
fig = plt.figure() 
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,6))

# sub plot 1 - Original signal
ax1.plot(wav_original)
ax1.set(title='Original signal', xlabel='time')

# sub plot 2 - Recovered signal
ax2.plot(wav_recovered)
ax2.set(title='recovered signal', xlabel='time')

# sub plot 3 - Recovered signal with phase
ax3.plot(wav_recovered_with_phase)
ax3.set(title='recovered signal with phase', xlabel='time')

plt.tight_layout()

#%% remove testing variables

del data_generator
del frame_size, maxFFTSize, sampling_rate, vad_threshold, wavfile1, wavfile2, sample, \
 stft, stft2, wav_original, wav_recovered, wav_recovered_with_phase, phase
 
#%% test phase

frame_size = 256
maxFFTSize = 129
sampling_rate = 8000
vad_threshold = 0.001
#Set wav files to combine
wavfile1 = "../../TIMIT_WAV/Train/DR1/FCJF0/SI1027.WAV" 
wavfile2 = "../../TIMIT_WAV/Train/DR1/MDPK0/SI552.WAV" 

# create instance of class
data_generator = DataGenerator() 
# Routine
sample = data_generator.CreateTrainingDataSpectrogram(wavfile1, wavfile2, sampling_rate, frame_size, maxFFTSize, vad_threshold)   

# Get phase from testing routine
ZmixedSeries = np.angle(sample['ZmixedSeries'])

# Get phase from testing routine
speech1_phase = sample['Speech1PhaseStd']
mixedphase = sample['SamplePhaseRaw']

print(np.max(ZmixedSeries))
print(np.min(ZmixedSeries))
print(np.max(mixedphase))
print(np.min(mixedphase))
print(np.max(speech1_phase))
print(np.min(speech1_phase))


#%% delete variables

del data_generator
del frame_size, maxFFTSize, sampling_rate, vad_threshold, wavfile1, wavfile2, sample, \
    ZmixedSeries, speech1_phase, mixedphase


#%% Test CreatePickleFiles function

#Directory of wav files
data_dir = "../../TIMIT_WAV/Train/DR8" 

#Location to save pickle files to
#Save pickle files to current working directory
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
# Number of frames per sample for batches (this will be the nimber of rows that are fed into each batch of the rnn)
frames_per_sample = 100

# create instance of class
data_generator = DataGenerator() 

# Start timer
start = time.time()

# Run CreatePickleFiles function
data_generator.CreatePickleFiles(filename, data_dir, sampling_rate, maxFFTSize, frame_size, vad_threshold, frames_per_sample)

# finish timer 
end = time.time()

print("Created test pickle file. Time elapsed: %f"%(end - start))

  
#%% Test training set

test = pickle.load(open(filename, 'rb'))

print("test length: %d"%len(test))

#%% Clear variables

del filename, sampling_rate, maxFFTSize, frame_size, vad_threshold, data_dir, frames_per_sample, end, start, test

#%% Create training set
#Use the folder DR1 to DR8 to create the training set

# Set random seed for repeatability
random.seed(9)

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
data_generator.CreatePickleForAllFolders("train", parentfolder, sampling_rate, maxFFTSize, frame_size, vad_threshold, endfolder, frames_per_sample, filefolder)

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

del parentfolder, sampling_rate, maxFFTSize, frame_size, vad_threshold, data_generator, frames_per_sample, endfolder, \
    filefolder, start, end


#%% Create testing set
#Use the folder DR1 to DR7 to create the training set

#Sampling rate
sampling_rate = 8000
#Maximum number of FFT points (NEFF)
maxFFTSize = 129
#Size of FFT frame
frame_size = 256

#Voice activity threshold (THRESHOLD)
#If TF bins are smaller than THRESHOLD then will be considered inactive
vad_threshold = 0.001
#Number of frames per sample for batches (this will be the nimber of rows that are fed into each batch of the rnn)
frames_per_sample = 100
 
#Set folder containing folders of wav files from TIMIT data set 
parentfolder = "../../TIMIT_WAV/TEST" 

#Set folder to save pickle files
#Folder is current working directory
filefolder = "../Data/"

#number of folders to loop through
endfolder = 7

# create instance of class
data_generator = DataGenerator() 

# Start timer
start = time.time()

# Run routine to create pickle files
data_generator.CreatePickleForAllFolders("test", parentfolder, sampling_rate, maxFFTSize, frame_size, vad_threshold, endfolder, frames_per_sample, filefolder, "X", 1)

# finish timer 
end = time.time()

print("Created testing pickle files. Time elapsed: %f"%(end - start))

#%% Test testing set

test1 = pickle.load(open("../Data/test1.pkl", 'rb'))
test2 = pickle.load(open("../Data/test2.pkl", 'rb'))
test3 = pickle.load(open("../Data/test3.pkl", 'rb'))

print("test1 length: %d"%len(test1))
print("test2 length: %d"%len(test2))
print("test3 length: %d"%len(test3))

   
    
     

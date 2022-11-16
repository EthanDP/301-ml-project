from genericpath import isfile
import os

import torch
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import torchaudio

# Parent class with methods used by all types of speech datasets
class SpeechDataset(Dataset):
    def __init__(self):
        self.LABELS = set(("calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"))
        super().__init__()

    def get_labels(self, file_dir, file_type):
        abs_path = os.path.join(os.getcwd, "..", file_dir)
        files = os.listdir(abs_path)

        labels = []

        for file in files:
            if os.path.isfile(file) and file_type in file:
                for label in self.LABELS:
                    if label in file:
                        labels.append((file, label))

       
class SpeechAudioDataset(SpeechDataset):
    def __init__(self, wav_dir, transform=None, target_transform=None):
        self.wav_labels = self.get_labels(wav_dir, ".wav")
        self.wav_dir = wav_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        wav_path = os.path.join(os.getcwd(), "..", self.wav_dir, self.wav_labels[idx][0])
        audio, samplerate = wavfile.read(wav_path)
        label = self.wav_labels[idx][1]
        if self.transform:
            audio = self.transform(audio, samplerate)
        if self.target_transform:
            label = self.target_transform(label)
        return (audio, samplerate), label


# Not implemented yet
class SpeechSpectrogramDataset(SpeechDataset):
    def __init__(self, signal, chunk_size):
        self.signal = signal
        self.chunk = chunk_size
        self.SAMPLING_RATE = 44100
        self.spectrogram_fs, self.spectrogram_t, self.spectrogram = spectrogram(self.signal, fs=self.SAMPLING_RATE, window='hanning')

    def mfc(self):
        '''
        Mel-Frequency Cepstrum: this method implements the algorithm to create and use Mel-Frequency Cepstrum analysis.

        inputs: 
            none, method runs on internal variables
        outputs:
            coefficients - an array of the Mel-frequency coefficients
        '''
        # algorithm: |F^-1[log(|F[x]|^2)]|^2
        #output_array = []
        #for time in range(self.spectrogram_t.shape[0]/self.chunk):
         #   squared = [bit**2 for bit in (np.abs(self.spectrogram[time]))]
         #   log = [np.log10(logbit) for logbit in squared]
         #   inverse_fft = np.fft.ifft(log)
         #   output = [out**2 for out in (np.abs(inverse_fft))]
         #   output_array.append(output)

        #mel_frequency = np.array(output_array)
        COEFFICIENTS_NUM = 40
        mfc_object = torchaudio.transforms.MFCC(sample_rate=self.SAMPLING_RATE, n_mfcc=COEFFICIENTS_NUM)
        coefficients = mfc_object(self.signal)

    def plot_spectrogram(self):
        '''
        Plots the spectrogram to help with visualization and debugging down the line.

        inputs:
            none, method runs on internal variables
        outputs:
            none, shows plot of spectrogram
        '''
        
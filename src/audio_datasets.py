from genericpath import isfile
import os

import torch
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from scipy.io import wavfile

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
    def __init__(self):
        pass
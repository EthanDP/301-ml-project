import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from audio_datasets import SpeechAudioDataset

def main():
    audio_dataset = SpeechAudioDataset("training_data/", None, None)

if __name__ == "__main__":
    main()
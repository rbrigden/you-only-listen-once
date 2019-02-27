import torch
import torch.nn as nn
import numpy as np
import training.speaker_verification.model as models
import torch.utils.data as data
import argparse
import librosa
import os
from torch.nn.utils.rnn import pad_sequence
import ray

@ray.remote
def process_with_mel(wav_path):
    y, sr = librosa.load(wav_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, n_mels=64)
    assert mel.shape[0] == 64
    return mel.T


class WavDataset(data.Dataset):

    def __init__(self, file_paths, get_label=None):
        super(WavDataset, self).__init__()

        if get_label is None:
            get_label = lambda x: x

        utterances = []
        labels = []
        for wav_path in file_paths:
            utterances.append(process_with_mel.remote(wav_path))
            labels.append(get_label(wav_path))

        self.utterances = ray.get(utterances)
        self.labels = labels

    def normalize(self, mean, std):
        mean, variance = mean, std
        norm_ = lambda u: (u - mean.reshape(1, -1)) / (std.reshape(1, -1))
        self.utterances = [norm_(u) for u in self.utterances]

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, item):
        return [torch.FloatTensor(self.utterances[item]), self.labels[item]]






import torch
import torch.nn as nn
import numpy as np
import training.speaker_classification.model as models
import torch.utils.data as data
import argparse
import librosa
import os
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from collections import defaultdict

class WavDataset(data.Dataset):

    def __init__(self, source_path):
        super(WavDataset, self).__init__()
        file_names = os.listdir(source_path)

        utterances = []
        labels = []
        for name in file_names:
            wav_path = os.path.join(source_path, name)
            utterances.append(process_with_mel(wav_path))
            labels.append(os.path.splitext(name)[0])

        self.utterances = utterances
        self.labels = labels

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, item):
        return [torch.FloatTensor(self.utterances[item]), self.labels[item]]


def sample_normalize(x):
    stats_path = "data/voxceleb/voxceleb_stats.npy"
    mean, variance, _ = np.load(stats_path)
    mean, variance = torch.FloatTensor(mean), torch.FloatTensor(variance)
    return (x - mean.view(1, -1)) / torch.sqrt(variance.view(1, -1))


def process_with_mel(wav_path):
    y, sr = librosa.load(wav_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, n_mels=64)
    assert mel.shape[0] == 64
    return mel


def collate(batch):
    data = [item[0].permute(1, 0) for item in batch]
    data = [sample_normalize(x) for x in data]
    target = [item[1] for item in batch]
    return [data, target]


class SpeakerEmbedInference:

    def __init__(self, model):
        self.model = model

    def _process_data_batch(self, data_batch):
        # pad the sequences, each seq must be (L, *)
        seq_lens = [len(x) for x in data_batch]
        seq_batch = pad_sequence(data_batch, batch_first=True)
        return seq_batch.unsqueeze(1), seq_lens

    def forward(self, x):
        self.model.eval()
        seq_batch, seq_lens = self._process_data_batch(x)
        with torch.no_grad():
            embeddings = self.model([seq_batch, seq_lens], em=True)
        return embeddings

    def load_params(self, checkpoint_path):
        cpd = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(cpd["model"])


def transform(embeddings):
    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    output = tsne.fit_transform(reduced_embeddings)
    return output

def plot_embeddings(embeddings, labels):

    embeddings2d = transform(embeddings)
    data_points = defaultdict(list)

    classes = dict()
    colors = []
    c = 0
    for l in labels:
        name, num = l.split('_')
        num = int(num)

        if name not in classes:
            classes[name] = c
            c += 1

        colors.append(classes[name])

    x, y = embeddings2d[:, 0].reshape(-1), embeddings2d[:, 1].reshaphe(-1)
    plt.scatter(x, y, c=np.array(colors))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str,
                        default='/tmp/demo_utterances',
                        help='directory with wavs')

    args = parser.parse_args()
    source_path = args.source

    model = models.SpeakerClassifier2d(1200)

    inference = SpeakerEmbedInference(model)

    dset = WavDataset(source_path)

    loader = data.DataLoader(dset, shuffle=False, batch_size=1, collate_fn=collate)

    embeddings = []
    for utterances, labels in loader:
        embedding_batch = inference.forward(utterances)
        embeddings.append(embedding_batch)
    embeddings = torch.cat(embeddings)

    plot_embeddings(embeddings.numpy(), labels)
    plt.show()
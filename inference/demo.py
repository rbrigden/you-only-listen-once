import torch
import numpy as np
import training.speaker_verification.model as models
import torch.utils.data as data
import argparse
import os
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import ray
from data.processing.audio import WavDataset
from collections import defaultdict

def collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


class SpeakerEmbedInference:

    def __init__(self, model):
        self.model = model

    def _process_data_batch(self, data_batch):
        # pad the sequences, each seq must be (L, *)
        seq_lens = [len(x) for x in data_batch]
        seq_batch = pad_sequence(data_batch, batch_first=True)
        return seq_batch.unsqueeze(1).cuda(), seq_lens

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
    pca = PCA(n_components=30)
    reduced_embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=5)
    output = tsne.fit_transform(reduced_embeddings)
    return output


def plot_embeddings(embeddings, labels):
    print(embeddings.shape)
    embeddings2d = transform(embeddings)


    class_dict = defaultdict(list)
    for i, name in enumerate(labels):
        class_dict[name].append(embeddings2d[i, :])



    for i, (name, es) in enumerate(class_dict.items()):
        es = np.stack(es)
        x, y = es[:, 0].reshape(-1), es[:, 1].reshape(-1)
        plt.scatter(x, y, label=str(name).capitalize(), cmap='rainbow')

    plt.legend(loc="upper left")


def get_label(file_path):
    name, _ = os.path.basename(file_path).split("_")
    return name


if __name__ == '__main__':
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str,
                        default='/tmp/demo_utterances',
                        help='directory with wavs')

    parser.add_argument("--param-path", type=str,
                        default='models/speaker_classification/simple_v2.pt',
                        help='directory with wavs')
    parser.add_argument("--stats-path", type=str,
                        default='stats.npy',
                        help='stats to normalize the data with')

    args = parser.parse_args()
    source_path = args.source
    names = os.listdir(source_path)
    file_paths = [os.path.join(source_path, name) for name in names]

    model = models.IdentifyAndEmbed(1200).cuda()
    inference = SpeakerEmbedInference(model)
    inference.load_params(args.param_path)

    dset = WavDataset(file_paths, get_label=get_label)
    if args.stats_path:
        mean, std = np.load(args.stats_path)
        dset.normalize(mean, std)

    loader = data.DataLoader(dset, shuffle=False, batch_size=32, collate_fn=collate)

    embeddings = []
    labels = []
    for utterance_batch, label_batch in loader:
        embedding_batch = inference.forward(utterance_batch)
        embeddings.append(embedding_batch)
        labels += label_batch
    embeddings = torch.cat(embeddings).cpu()

    plot_embeddings(embeddings.numpy(), labels)
    plt.savefig("fig.png")

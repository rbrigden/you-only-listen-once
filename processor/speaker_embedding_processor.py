import torch
import numpy as np
import inference
import training.speaker_verification.model as models
from torch.nn.utils.rnn import pad_sequence
import gin

@gin.configurable
class SpeakerEmbeddingProcessor:

    def __init__(self, model_cls, checkpoint_path, use_gpu=False):
        self.embedding_model = model_cls()
        self.inference_engine = SpeakerEmbeddingInference(self.embedding_model, use_gpu=use_gpu)
        self.inference_engine.load_params(checkpoint_path)

    def forward(self, spect_batch):
        return self.inference_engine.forward(spect_batch)

    def __call__(self, spect_batch):
        return self.forward(spect_batch)


def normalize(x, mean, variance):
    return (x - mean.reshape(1, -1)) / np.sqrt(variance.reshape(1, -1))

@gin.configurable
class SpeakerEmbeddingInference:

    def __init__(self, model, stats_path, use_gpu=False):
        self.mean, self.variance, _ = np.load(stats_path)
        self.model = model.cuda() if use_gpu else model
        self.use_gpu = use_gpu

    def forward(self, utterance_batch):
        """ Compute utterances
        :param utterance_batch: list of (T, F)
        :return: embedding matrix (N, D)
        """
        self.model.eval()
        utterance_batch = [normalize(u, self.mean, self.variance) for u in utterance_batch]
        seq_batch, seq_lens = process_data_batch(utterance_batch, mode="wrap")
        seq_batch = seq_batch.cuda() if self.use_gpu else seq_batch
        with torch.no_grad():
            embeddings = self.model([seq_batch, seq_lens], em=True)
        return embeddings.cpu()

    def load_params(self, checkpoint_path):
        cpd = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(cpd["model"])


def process_data_batch(data_batch, mode='zeros'):
    # pad the sequences, each seq must be (L, *)
    seq_lens = [len(x) for x in data_batch]
    max_len = max(seq_lens)
    pad_widths = [max_len - l for l in seq_lens]

    if mode == 'zeros':
        data_batch = sorted(data_batch, key=lambda s: s.shape[0], reverse=True)
        seq_lens = sorted(seq_lens, reverse=True)
        seq_batch = pad_sequence(data_batch, batch_first=True)
    elif mode == 'wrap':
        seq_batch = torch.stack(
            [torch.FloatTensor(np.pad(x, pad_width=((0, w), (0, 0)), mode='wrap')) for w, x in
             zip(pad_widths, data_batch)])
    else:
        raise ValueError("Invalid mode specified")

    return seq_batch.unsqueeze(1), seq_lens

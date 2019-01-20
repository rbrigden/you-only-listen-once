import torch
import torch.utils.data.dataset as dataset
import ray
import numpy as np
from torch.utils.data.sampler import Sampler
from torch._six import int_classes as _int_classes

FEATS = 64


class FixedWidthUtteranceSample(dataset.Dataset):
    def __init__(self, utterances, chunk_size):
        super(FixedWidthUtteranceSample, self).__init__()
        self.utterances = utterances
        self.chunk_size = chunk_size

    def __len__(self):
        return self.utterances.shape[0]

    def __getitem__(self, item):
        u = self.utterances[item]
        length = u.shape[0]
        if length > self.chunk_size:
            start_idx = np.random.randint(0, length - self.chunk_size + 1)
            feats = torch.FloatTensor(u[start_idx:start_idx + self.chunk_size])
        else:
            pad_width = self.chunk_size - length
            feats = torch.FloatTensor(np.pad(u, ((0, pad_width), (0, 0)), mode='wrap'))
        return feats.unsqueeze(0)


class FixedWidthUtteranceSampleWithSpeakers(dataset.Dataset):
    def __init__(self, utterances, labels, chunk_size):
        super(FixedWidthUtteranceSampleWithSpeakers, self).__init__()
        self.utterances = utterances
        self.labels = labels
        self.chunk_size = chunk_size

    def __len__(self):
        return self.utterances.shape[0]

    def __getitem__(self, item):
        u = self.utterances[item]
        length = u.shape[0]
        if length > self.chunk_size:
            start_idx = np.random.randint(0, length - self.chunk_size + 1)
            feats = torch.FloatTensor(u[start_idx:start_idx + self.chunk_size])
        else:
            pad_width = self.chunk_size - length
            feats = torch.FloatTensor(np.pad(u, ((0, pad_width), (0, 0)), mode='wrap'))
        label = self.labels[item]
        return (feats.unsqueeze(0), label)


class RandomSampleWithReplacement(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        data_len = len(self.data_source)
        return iter(torch.randint(0, data_len, size=(data_len,)).tolist())

    def __len__(self):
        return len(self.data_source)


class ChunkedTrainingSet(dataset.Dataset):
    def __init__(self, utterances, labels, chunk_size):
        super(ChunkedTrainingSet, self).__init__()
        self.utterances = utterances
        self.labels = labels
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, item):
        u = self.utterances[item]
        length = u.shape[0]
        if length > self.chunk_size:
            start_idx = np.random.randint(0, length - self.chunk_size + 1)
            feats = torch.FloatTensor(u[start_idx:start_idx + self.chunk_size])
        else:
            pad_width = self.chunk_size - length
            feats = torch.FloatTensor(np.pad(u, ((0, pad_width), (0, 0)), mode='wrap'))
        label = self.labels[item]
        return (torch.transpose(feats, 0, 1), label)


@ray.remote
def wrap_to_max(u, max):
    length = u.shape[0]
    pad_width = max - length
    return np.pad(u, ((0, pad_width), (0, 0)), mode='wrap')


@ray.remote
def random_chunk_or_wrap(u, thresh):
    length = u.shape[0]
    if length < thresh:  # wrap
        pad_width = thresh - length
        processed_u = np.pad(u, ((0, pad_width), (0, 0)), mode='wrap')
    elif length > thresh:  # trim to thresh
        start_idx = np.random.randint(0, length - thresh + 1)
        processed_u = u[:thresh]
    else:
        processed_u = u
    return processed_u.reshape(1, thresh, FEATS)


def process_chunks(utterances, threshold):
    chunks = ray.get([random_chunk_or_wrap.remote(u, threshold) for u in utterances])
    return np.vstack(chunks).astype(np.float32)


def dev_proccess(utterances):
    max_length = max([x.shape[0] for x in utterances])
    processed_u = ray.get([wrap_to_max.remote(u, max_length) for u in utterances])
    return torch.FloatTensor(np.vstack(processed_u))


def process_utterances(utterances, threshold=None):
    lengths = [x.shape[0] for x in utterances]
    thresh = max(lengths) if threshold is None else threshold
    processed_utterances, masks = zip(*[pad_or_trim_utterance.remote(u, thresh) for u in utterances])
    processed_masks = np.vstack([ray.get(mask) for mask in masks])
    processed_utterances = np.vstack([ray.get(u) for u in processed_utterances])
    return processed_utterances.astype(np.float32), processed_masks.astype(np.float32)


class SiameseFixedWidthUtteranceSampleWithSpeakers(dataset.Dataset):
    def __init__(self, utterances, labels, chunk_size):
        super(SiameseFixedWidthUtteranceSampleWithSpeakers, self).__init__()
        self.utterances = utterances
        self.labels = labels
        self.chunk_size = chunk_size

    def __len__(self):
        return self.utterances.shape[0]

    def process_feats(self, u):
        length = u.shape[0]
        if length > self.chunk_size:
            start_idx = np.random.randint(0, length - self.chunk_size + 1)
            feats = torch.FloatTensor(u[start_idx:start_idx + self.chunk_size])
        else:
            pad_width = self.chunk_size - length
            feats = torch.FloatTensor(np.pad(u, ((0, pad_width), (0, 0)), mode='wrap'))
        return feats

    def __getitem__(self, item):
        i0, i1 = item
        u0 = self.utterances[i0]
        u1 = self.utterances[i1]
        feats0 = self.process_feats(u0)
        feats1 = self.process_feats(u1)
        labels0 = self.labels[i0]
        labels1 = self.labels[i1]
        return (feats0.unsqueeze(0), feats1.unsqueeze(0), labels0, labels1)


class ContrastiveBatchSampler(Sampler):
    def __init__(self, labels, nspeakers, batch_size):
        speaker_maps = []
        self.batch_size = batch_size
        self.nspeakers = nspeakers
        self.labels = labels
        for i in range(nspeakers):
            speaker_idxs = [j for j, l in enumerate(labels) if l == i]
            np.random.shuffle(speaker_idxs)
            speaker_idxs = [(speaker_idxs[k], speaker_idxs[k + 1]) for k in range(0, len(speaker_idxs) - 1, 2)]
            speaker_maps += speaker_idxs
        self.speaker_maps = speaker_maps
        self.num_batches = (len(labels) + self.batch_size - 1) // batch_size

        self.positive_per_batch = len(speaker_maps) // self.num_batches

    def build_batch(self):
        num_positive = self.positive_per_batch
        num_negative = self.batch_size - num_positive

        positive_samples = [self.speaker_maps.pop(np.random.randint(len(self.speaker_maps))) for _ in
                            range(num_positive)]
        negative_samples = zip(np.random.randint(len(self.labels), size=num_negative),
                               np.random.randint(len(self.labels), size=num_negative))

        return list(positive_samples) + list(negative_samples)

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.build_batch()

    def __len__(self):
        return (len(self.labels) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    labels = np.random.randint(10, size=126)
    loader = ContrastiveBatchSampler(labels, 876, 10)

    print(len(loader))
    c = 0
    for b in loader:
        c += 1
    print(c)

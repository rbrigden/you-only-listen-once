import torch
import numpy as np
import torch.utils.data as data


class Siamese(data.Dataset):

    def __init__(self, dset):
        super(Siamese, self).__init__()
        self.dset = dset

    def __len__(self):
        return self.dset

    def __getitem__(self, item):
        i0, i1 = item
        u0, l0 = self.dset[i0]
        u1, l1 = self.dset[i1]
        return [u0, u1, l0, l1]


class ContrastiveSampler(data.BatchSampler):

    def __init__(self, labels, nspeakers, batch_size):
        speaker_maps = []
        self.batch_size = batch_size
        self.nspeakers = nspeakers
        self.labels = labels
        for s in range(nspeakers):
            speaker_idxs = [idx for idx, label in enumerate(labels) if label == s]
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

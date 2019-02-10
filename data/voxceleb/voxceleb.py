import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import re
import ray
import copy


@ray.remote
def load_file(path):
    return np.load(path)


def get_all_data_file_paths(processed_root):
    paths = []
    for root, _, filenames in os.walk(processed_root):
        for filename in filenames:
            processed_file_path = os.path.join(root, filename)
            paths.append(processed_file_path)
    return paths


def speaker_label_from_path(path):
    p = re.compile("id\d+")
    res = p.search(path)
    return int(res.group(0)[2:]) - 10001


def load_file_paths_and_labels(file_paths, speakers):
    # filter out samples from other speakers
    sids = [speaker_label_from_path(path) for path in file_paths]

    valid_paths_and_labels = [(path, sid) for sid, path in zip(sids, file_paths) if sid in speakers]

    valid_paths, labels = zip(*valid_paths_and_labels)
    return valid_paths, labels


def load_files(file_paths):
    return ray.get([load_file.remote(path) for path in file_paths])


class VoxcelebID(Dataset):
    """ Utterances with speaker labels """

    def __init__(self, processed_root, speakers, preload=False):
        super(VoxcelebID, self).__init__()
        self.speakers = speakers
        self.processed_root = processed_root
        all_paths = get_all_data_file_paths(processed_root)
        self.utterance_paths, self.utterance_labels = load_file_paths_and_labels(all_paths, speakers)

        self.utterances = load_files(self.utterance_paths) if preload else None
        self.preload = preload

    def __getitem__(self, item):
        if self.preload:
            u = self.utterances[item]
        else:
            u = np.load(self.utterance_paths[item])
        l = self.utterance_labels[item]
        return [torch.FloatTensor(u), np.int64(l)]

    def __len__(self):
        return len(self.utterance_labels)

    @classmethod
    def create_split(cls, processed_root, speakers, split=0.8, shuffle=True):
        """ Create two datasets that are a split of other
            first returned has split and the other has 1-split
            number of samples.

        """

        dset1 = cls(processed_root, speakers)
        dset2 = copy.deepcopy(dset1)

        order = np.arange(len(dset1))
        if shuffle:
            np.random.shuffle(order)

        utterance_paths = [dset1.utterance_paths[i] for i in order]
        utterance_labels = [dset1.utterance_labels[i] for i in order]

        split_point = int(split * len(dset1))

        dset1.utterance_paths = utterance_paths[:split_point]
        dset1.utterance_labels = utterance_labels[:split_point]

        dset2.utterance_paths = utterance_paths[split_point:]
        dset2.utterance_labels = utterance_labels[split_point:]
        return dset1, dset2


def voxceleb_collate(batch):
    data = [item[0].permute(1, 0) for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def parse_verification_file(veri_file_path, processed_root):
    enrol_paths, test_paths, labels = [], [], []
    with open(veri_file_path, 'r') as f:
        for line in f.readlines():
            label, enrol_sample_path, test_sample_path = line.strip().split(" ")
            label = np.int64(label)

            # Add full path and fix extension
            enrol_sample_path = "{}.npy".format(os.path.join(processed_root, os.path.splitext(enrol_sample_path)[0]))
            test_sample_path = "{}.npy".format(os.path.join(processed_root, os.path.splitext(test_sample_path)[0]))

            enrol_paths.append(enrol_sample_path)
            test_paths.append(test_sample_path)
            labels.append(label)
    return enrol_paths, test_paths, labels


class VoxcelebVerification(Dataset):
    """ Elegantly load verification utterances without redundancy
    """

    def __init__(self, paths):
        """ DONT use this init. alway use build """
        self.sample_paths = paths
        self.distinct_paths = sorted(list(set(paths)))
        self.sample_idxs = [self.distinct_paths.index(path) for path in paths]

    def __len__(self):
        return len(self.distinct_paths)

    def __getitem__(self, item):
        sample_path = self.distinct_paths[item]
        utterance = np.load(sample_path)
        return [torch.FloatTensor(utterance)]

    @classmethod
    def build(cls, veri_file_path, processed_test_root):
        enrol_paths, test_paths, labels = parse_verification_file(veri_file_path, processed_test_root)
        enrol_set = cls(enrol_paths)
        test_set = cls(test_paths)
        return enrol_set, test_set, labels

def voxceleb_veri_collate(batch):
    utterances = [item[0].permute(1, 0) for item in batch]
    return [utterances]


if __name__ == "__main__":
    # Only load from first 10 speakers
    # speakers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # dset = VoxcelebID("/home/rbrigden/voxceleb/processed", speakers, preload=False)
    # loader = DataLoader(dset, batch_size=32, shuffle=True, num_workers=8, collate_fn=voxceleb_collate)
    #
    #
    # for idx, (utterance_batch, label_batch) in enumerate(loader):
    #     print("Batch: {}".format(idx))
    #     for u in utterance_batch:
    #         print(u.shape)
    #     print()

    enrol, test, labels = parse_verification_file("data/voxceleb/veri_test.txt",
                                                  "/home/rbrigden/voxceleb/test/processed")
    print()

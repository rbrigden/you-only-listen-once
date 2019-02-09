import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import re
import ray

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

def voxceleb_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]



if __name__ == "__main__":
    # Only load from first 10 speakers
    speakers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dset = VoxcelebID("/home/rbrigden/voxceleb/processed", speakers, preload=False)
    loader = DataLoader(dset, batch_size=32, shuffle=True, num_workers=8, collate_fn=voxceleb_collate)
    
    
    for idx, (utterance_batch, label_batch) in enumerate(loader):
        print("Batch: {}".format(idx)) 
        for u in utterance_batch:
            print(u.shape)
        print()
        

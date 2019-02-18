import numpy as np
import argparse
from data.processing.audio import WavDataset
import torch.utils.data as data
import os
import ray

def welford_update(existingAggregate, newValue):
    """ https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance """
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)


def finalize(existingAggregate):
    """ https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance """
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1))
    if count < 2:
        return float('nan')
    else:
        return (mean, variance, sampleVariance)

def collate(batch):
    data = [item[0] for item in batch]
    return [data, ]

def compute_statistics(file_paths, feats=64):
    dset = WavDataset(file_paths)
    loader = data.DataLoader(dset, batch_size=100, shuffle=False, num_workers=10, collate_fn=collate)
    aggregate = (0, np.zeros(feats,), np.zeros(feats,))

    num_batches = len(dset) // 100


    for idx, (utterance_batch,) in enumerate(loader):
        for u in utterance_batch:
            for t in range(u.shape[0]):
                aggregate = welford_update(aggregate, u[t].numpy().reshape((feats,)))
        print("idx = {} / {}".format(idx, num_batches))
    mean, variance, sampleVariance = finalize(aggregate)
    return [mean, np.sqrt(variance)]

if __name__ == '__main__':
    ray.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str,
                        default='/tmp/demo_utterances',
                        help='directory with wavs')
    parser.add_argument("--dest", type=str,
                        default='stats.npy',
                        help='path to stats out')

    args = parser.parse_args()
    source_path = args.source

    names = os.listdir(source_path)
    file_paths = [os.path.join(source_path, name) for name in names]

    stats = compute_statistics(file_paths)
    np.save(args.dest, stats)


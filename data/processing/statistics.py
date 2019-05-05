import numpy as np
import argparse
from data.processing.audio import WavDataset, ProcessedDataset
import torch.utils.data as data
import os
import ray
from datetime import datetime
import pickle as pkl


@ray.remote
def spect_stats(spect_path):
    spect = np.load(spect_path)
    return [np.mean(spect, axis=0), 1, np.var(spect, axis=0)]


@ray.remote
def merge_stats(a, b):
    avg_a, count_a, var_a = a
    avg_b, count_b, var_b = b

    delta = avg_b - avg_a
    m_a = var_a * (count_a - 1)
    m_b = var_b * (count_b - 1)

    count = count_a + count_b
    avg = avg_a + delta * (count_b / count)
    M2 = m_a + m_b + delta ** 2 * count_a * count_b / (count_a + count_b)
    var = M2 / (count_a + count_b - 1)
    return [avg, count, var]


def tree_reduce(data):
    while len(data) > 1:
        a = data.pop(0)
        b = data.pop(0)
        data.append(merge_stats.remote(a, b))
    return ray.get(data)[0]


if __name__ == '__main__':
    ray.init(num_cpus=10)

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str,
                        default='/tmp/demo_utterances',
                        help='directory with wavs')
    parser.add_argument("--dest", type=str,
                        default='data/voxceleb/voxceleb_stats',
                        help='path to stats out')

    args = parser.parse_args()
    source_path = args.source

    file_paths = [os.path.join(root, name) for root, subdirs, names in os.walk(source_path) for name in names]
    print("Launched tasks")
    data = [spect_stats.remote(fp) for fp in file_paths]
    dset_mean, dset_count, dset_var = tree_reduce(data)

    stats = {
        "mean": dset_mean,
        "count": dset_count,
        "var": dset_var
    }

    now = datetime.now()
    tstamp = now.strftime("m%md%dy%Yh%Hm%Ms%S")

    stats_path = "{}_{}.pkl".format(args.dest, tstamp)
    with open(stats_path, 'wb') as f:
        pkl.dump(stats, f)

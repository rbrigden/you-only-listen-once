import os
import numpy as np
import scipy.io.wavfile as wav
import argparse
import time
import ray
import speechpy
from datetime import datetime
import pickle as pkl


def procces_wav(wav_path):
    fs, signal = wav.read(wav_path)
    spect = speechpy.feature.lmfe(signal, sampling_frequency=fs, num_filters=64)
    return spect

@ray.remote
def process_with_mel(wav_path, out_path, target_sample_rate):
    spect = procces_wav(wav_path)
    np.save(out_path, spect)
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

        print("Merge: {} and {}".format(a, b))

        data.append(merge_stats.remote(a, b))
    return ray.get(data)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str,
                        default='/home/rbrigden/voxceleb/wav')
    parser.add_argument("--dest", type=str,
                        default='/home/rbrigden/voxceleb/processed_stft')
    parser.add_argument("--mode", type=str, default="mel")
    parser.add_argument("--stats-path", type=str, default="data/voxceleb/voxceleb_stats")

    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--compute-stats", action='store_true', default=False)

    args = parser.parse_args()

    source_path = args.source
    dest_path = args.dest

    ray.init(num_cpus=18)

    if not os.path.exists(source_path):
        raise ValueError("Source path is invalid")

    if not os.path.exists(dest_path):
        raise ValueError("Dest path is invalid")

    start = time.time()
    task_args = []
    for root, _, filenames in os.walk(source_path):
        for filename in filenames:
            wav_file_path = os.path.join(root, filename)
            out_name = os.path.splitext(filename)[0]
            # base_name = root.split("/")[-2:]
            out_path = os.path.join(*([dest_path]+[out_name]))

            task_args.append((wav_file_path, out_path, args.sample_rate))

    print(len(task_args))
    tasks = [process_with_mel.remote(*x) for x in task_args]
    print("Launched Tasks")

    if args.compute_stats:
        dset_mean, dset_count, dset_var = tree_reduce(tasks)
        stats = {
            "mean": dset_mean,
            "count": dset_count,
            "var": dset_var
        }

        now = datetime.now()
        tstamp = now.strftime("m%md%dy%Yh%Hm%Ms%S")

        stats_path = "{}_{}.pkl".format(args.stats_path, tstamp)
        with open(stats_path, 'wb') as f:
            pkl.dump(stats, f)
    else:
        ray.wait(tasks, num_returns=len(tasks))

    end = time.time()
    print("Finished in {} secs".format(end - start))

import os
import numpy as np
import librosa
import librosa.display
import argparse
import time
import ray
import scipy.signal

@ray.remote
def process_with_mel(wav_path, out_path, target_sr=None):
    data, sr = librosa.load(wav_path)
    if target_sr and target_sr != sr:
        librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    mel = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=2048, n_mels=64)
    assert mel.shape[0] == 64
    np.save(out_path, mel)

@ray.remote
def process_with_stft(wav_path, out_path, target_sr=16000):
    data, sr = librosa.load(wav_path)
    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    spect = librosa.stft(y=data, n_fft=1024, win_length=400, hop_length=160)
    spect = np.abs(spect)
    np.save(out_path, spect)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str,
                        default='/home/rbrigden/nist-sre/raw/wav')
    parser.add_argument("--dest", type=str,
                        default='/home/rbrigden/nist-sre/raw/processed')
    parser.add_argument("--sample-rate", type=int, default=None)
    
    args = parser.parse_args()

    source_path = args.source
    dest_path = args.dest

    ray.init()

    if not os.path.exists(source_path):
        raise ValueError("Source path is invalid")

    if not os.path.exists(dest_path):
        raise ValueError("Dest path is invalid")

    start = time.time()
    object_ids = []

    for wav_file_name in os.listdir(args.source):
        wav_file_path = os.path.join(args.source, wav_file_name)
        npy_file_path = os.path.join(args.dest, os.path.splitext(wav_file_name)[0])
        task = process_with_stft.remote(wav_file_path, npy_file_path, args.sample_rate)
        object_ids.append(task)

    ray.wait(object_ids, num_returns=len(object_ids))
    end = time.time()

    print("Finished in {} secs".format(end - start))

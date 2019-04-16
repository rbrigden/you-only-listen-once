import os
import numpy as np
import librosa
import librosa.display
import argparse
import time
import ray


@ray.remote
def process_with_mfcc(wav_path, out_path, sample_rate):
    y, sr = librosa.load(wav_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, dct_type=2, n_mfcc=64, norm='ortho')
    np.save(out_path, mfcc)

@ray.remote
def process_with_mel(wav_path, out_path, sample_rate):
    # TODO: Standardize sr across dataset
    y, sr = librosa.load(wav_path)
    print(sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, n_mels=64)
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
                        default='/home/rbrigden/voxceleb/wav')
    parser.add_argument("--dest", type=str,
                        default='/home/rbrigden/voxceleb/processed_stft')
    parser.add_argument("--mode", type=str,
                        default="mfcc")
    parser.add_argument("--sample-rate", type=int, default=22050)
    
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
    for root, _, filenames in os.walk(source_path):
        for filename in filenames:
            wav_file_path = os.path.join(root, filename)
            out_name = os.path.splitext(filename)[0]
            base_name = root.split("/")[-2:]
            out_path = os.path.join(*([dest_path]+base_name+[out_name]))
            
            if args.mode == "mfcc":
                task = process_with_mfcc.remote(wav_file_path, out_path, args.sample_rate)
            elif args.mode == "mel":
                task = process_with_mel.remote(wav_file_path, out_path, args.sample_rate)
            elif args.mode == "stft":
                task = process_with_stft.remote(wav_file_path, out_path, args.sample_rate)
            else:
                raise ValueError("Must specify valid mode")
            object_ids.append(task)

    ray.wait(object_ids, num_returns=len(object_ids))
    end = time.time()

    print("Finished in {} secs".format(end - start))

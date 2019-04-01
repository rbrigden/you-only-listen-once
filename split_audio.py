from pydub import AudioSegment
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None)
    parser.add_argument("--n", type=int, default=4)

    args = parser.parse_args()
    newAudio = AudioSegment.from_wav(args.src)
    duration_ms = newAudio.duration_seconds * 1000
    splits = np.cumsum([0] + [int(duration_ms) // args.n for _ in range(args.n)])
    segments = [newAudio[splits[i]:splits[i+1]] for i in range(len(splits) - 1)]
    path = os.path.splitext(args.src)[0]
    outpaths = ["{}{}.wav".format(path, i) for i in range(args.n)]

    for outpath, segment in zip(outpaths, segments):
        segment.export(outpath, format="wav")
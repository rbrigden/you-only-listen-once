from __future__ import print_function
import os
import sys
import numpy as np
from tqdm import tqdm



if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[2] not in list(map(str, range(1, 7))) + ["dev", "test"]:
        print("Usage: python2", sys.argv[0], "<path to npz files>", "<chunk among {1, 2, .., 6, dev, test}>")
        exit(0)
    
    if sys.version_info.major != 2:
        print("\nWarning: We strongly recommend using Python 2 when running this script.\n"
              "Otherwise, np.savez() may write to disk somewhat larger npz files (or fail, for some installations).\n")
        exit(0)

    path, part = sys.argv[1], sys.argv[2]
    input_path = os.path.join(path, part + ".npz")
    output_path = os.path.join(path, part + ".preprocessed.npz")

    npz = np.load(input_path, encoding='latin1')

    if part == "dev":
        np.savez(output_path, enrol=bulk_VAD(npz['enrol']), test=bulk_VAD(npz['test']), trials=npz['trials'],
                 labels=npz['labels'])

    elif part == "test":
        np.savez(output_path, enrol=bulk_VAD(npz['enrol']), test=bulk_VAD(npz['test']), trials=npz['trials'])

    else:
        np.savez(output_path, feats=bulk_VAD(npz['feats']), targets=npz['targets'])

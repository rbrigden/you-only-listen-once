import os
import numpy as np
import librosa
import librosa.display
import argparse
import time
import ray

@ray.remote
def process_with_mfcc(wav_path, out_path):
  y, sr = librosa.load(wav_path)
  mfcc = librosa.feature.mfcc(y = y, sr = sr, dct_type = 2, norm = 'ortho')
  np.save(out_path, mfcc)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=str, default='/home/rbrigden/voxceleb/wav')
  parser.add_argument("--dest", type=str, default='/home/rbrigden/voxceleb/processed')
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
      base_name  = root.split("/")[-2:]
      out_path = os.path.join(*([dest_path]+base_name+[out_name]))
      #print("Read from: {}".format(wav_file_path))
      #print("Write to: {}".format(out_path)) 
      object_ids.append(process_with_mfcc.remote(wav_file_path, out_path))
  
  ray.wait(object_ids, num_returns=len(object_ids))
  end = time.time()
  
  print("Finished in {} secs".format(end - start))































import os
import numpy as np
import librosa
import librosa.display 



root_filepath = 'voxceleb/wav/'
dest_filepath = 'voxceleb/processed/'

# id10001/1zcIwhmdeo4/00002.wav'

def get_mfcc(filepath):
	y, sr = librosa.load(filepath)
	mfcc = (librosa.feature.mfcc(y = y, sr = sr, dct_type = 2, norm = 'ortho'))
	return mfcc

for root, directories, filenames in os.walk(root_filepath):
	for filename in filenames: 
		wav_FilePath = os.path.join(root,filename)
		mfcc_wav = get_mfcc(wav_FilePath)
		dest = dest_filepath + wav_FilePath 
		np.save(dest, mfcc_wav)


































# Packages reqd: pydub, ffmpeg

# pydub - pip install pydub

# ffmpeg:
# sudo add-apt-repository ppa:kirillshkrogalev/ffmpeg-next
# sudo apt-get update
# sudo apt-get install ffmpeg

## Load the m4a files (in M4a_files.tar.gz)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default="internal_data")
args = parser.parse_args()

## Delete unwanted files (here: Tapping files)


## Converting to wav
# Using pydub

# Convert all file extensions to m4a (if required)

import os, sys

folder = args.src
for filename in os.listdir(folder):
    infilename = os.path.join(folder, filename)
    if not os.path.isfile(infilename): continue
    oldbase = os.path.splitext(filename)
    newname = infilename.replace('.tmp', '.m4a')
    output = os.rename(infilename, newname)

# Convert m4a extension files to wav extension files

import os

from pydub import AudioSegment

formats_to_convert = ['.m4a']

for (dirpath, dirnames, filenames) in os.walk(args.src):
    for filename in filenames:
        if filename.endswith(tuple(formats_to_convert)):

            filepath = dirpath + '/' + filename
            (path, file_extension) = os.path.splitext(filepath)
            file_extension_final = file_extension.replace('.', '')
            try:
                track = AudioSegment.from_file(filepath,
                                               file_extension_final)
                wav_filename = filename.replace(file_extension_final, 'wav')
                wav_path = dirpath + '/' + wav_filename
                print('CONVERTING: ' + str(filepath))
                file_handle = track.export(wav_path, format='wav')
                os.remove(filepath)
            except:
                print("ERROR CONVERTING " + str(filepath))


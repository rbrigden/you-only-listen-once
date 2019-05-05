import soundfile as sf
import io
import logging
import gin
import librosa
from pydub import AudioSegment
import numpy as np
from io import BytesIO
import speechpy
from scipy.io import wavfile

@gin.configurable
class AudioProcessor:

    def __init__(self,
                 window_size=2048,
                 num_feats=64,
                 sample_rate=None):
        self.window_size = window_size
        self.num_feats = num_feats
        self.default_sample_rate = sample_rate
        self.logger = logging.getLogger('audioProcessor')

    def forward(self, audio_bytes, split=1):
        # TODO: Add split logic

        audio_stream = io.BytesIO(audio_bytes)

        all_mels = []

        if split > 1:
            newAudio = AudioSegment.from_wav(audio_stream)
            duration_ms = newAudio.duration_seconds * 1000
            splits = np.cumsum([0] + [int(duration_ms) // split for _ in range(split)])
            segments = [newAudio[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]

            for segment in segments:
                segment.export("temp.wav", format='wav')
                with open("temp.wav", 'rb') as f:
                    data, source_sample_rate = sf.read(f, always_2d=True)
                    data = data[:, 0]
                self.logger.info("Source sample rate is {}".format(source_sample_rate))
                mel = speechpy.feature.lmfe(data, sampling_frequency=source_sample_rate, num_filters=64)
                mel = mel - np.mean(mel, axis=0, dtype=np.float64)
                all_mels.append(mel)
        else:
            data, source_sample_rate = sf.read(audio_stream, always_2d=True)
            data = data[:, 0]
            self.logger.info("Source sample rate is {}".format(source_sample_rate))
            mel = speechpy.feature.lmfe(data, sampling_frequency=source_sample_rate, num_filters=64)
            mel = mel - np.mean(mel, axis=0, dtype=np.float64)
            all_mels.append(mel)

        return all_mels, source_sample_rate, data


    def from_file(self, path):
        source_sample_rate, data = wavfile.read(path)
        if len(data.shape) > 1:
            data = data[:, 0]
        self.logger.info("Source sample rate is {}".format(source_sample_rate))
        mel = speechpy.feature.lmfe(data, sampling_frequency=source_sample_rate, num_filters=64)
        mel = mel - np.mean(mel, axis=0, dtype=np.float64)
        return [mel]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


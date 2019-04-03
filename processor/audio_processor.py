import soundfile as sf
import io
import logging
import gin
import librosa
from pydub import AudioSegment
import numpy as np
from io import BytesIO

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
                    data, source_sample_rate = sf.read(f)
                self.logger.info("Source sample rate is {}".format(source_sample_rate))

                if self.default_sample_rate is not None:
                    target_sample_rate = self.default_sample_rate
                    data = librosa.resample(data, orig_sr=source_sample_rate, target_sr=target_sample_rate)
                else:
                    target_sample_rate = source_sample_rate

                self.logger.info("Target sample rate is {}".format(target_sample_rate))

                mel = librosa.feature.melspectrogram(y=data, sr=target_sample_rate, n_fft=self.window_size, n_mels=self.num_feats)
                all_mels.append(mel.T)
        else:
            data, source_sample_rate = sf.read(audio_stream)
            self.logger.info("Source sample rate is {}".format(source_sample_rate))

            if self.default_sample_rate is not None:
                target_sample_rate = self.default_sample_rate
                data = librosa.resample(data, orig_sr=source_sample_rate, target_sr=target_sample_rate)
            else:
                target_sample_rate = source_sample_rate

            self.logger.info("Target sample rate is {}".format(target_sample_rate))

            mel = librosa.feature.melspectrogram(y=data, sr=target_sample_rate, n_fft=self.window_size,
                                                 n_mels=self.num_feats)
            all_mels.append(mel.T)


        return all_mels

    def from_file(self, path):
        data, source_sample_rate = librosa.load(path)

        self.logger.info("Source sample rate is {}".format(source_sample_rate))

        if self.default_sample_rate is not None:
            target_sample_rate = self.default_sample_rate
            data = librosa.resample(data, orig_sr=source_sample_rate, target_sr=target_sample_rate)
        else:
            target_sample_rate = source_sample_rate

        self.logger.info("Target sample rate is {}".format(target_sample_rate))

        mel = librosa.feature.melspectrogram(y=data, sr=target_sample_rate, n_fft=self.window_size, n_mels=self.num_feats)
        return [mel.T]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


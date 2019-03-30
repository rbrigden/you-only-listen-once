import soundfile as sf
import librosa
import io

class AudioProcessor:

    def __init__(self,
                 window_size=2048,
                 num_feats=64,
                 sample_rate=None):
        self.window_size = window_size
        self.num_feats = num_feats
        self.default_sample_rate = sample_rate

    def forward(self, audio_bytes, split=1):
        # TODO: Add split logic

        audio_stream = io.BytesIO(audio_bytes)
        data, sample_rate = sf.read(audio_stream)

        if self.default_sample_rate is not None:
            sample_rate = self.default_sample_rate

        mel = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_fft=self.window_size, n_mels=self.num_feats)
        return [mel.T]

    def __call__(self, x):
        return self.forward(x)


import gin
import numpy as np
from presence_detection.speech_rec import SpeechRec
from presence_detection.fb import PresenceScore


class PresenceDetector:

    def __init__(self):
        gin.parse_config_file("presence_detection/config/presence_detection.gin")
        self.speech_rec_model = SpeechRec()
        self.presence_model = PresenceScore()

    def forward(self, ground_truth, audio, fs, threshold):
        log_probs = self.speech_rec_model.forward(audio, fs, display_chars=False)
        chars = self.speech_rec_model.alphabet._label_to_str + ["-"]
        score = self.presence_model.forward(ground_truth, log_probs, chars)
        print(score)
        return score > threshold


if __name__ == '__main__':
    from scipy.io import wavfile as wav

    sr = PresenceDetector()

    fs, audio = wav.read("/home/rbrigden/demo.wav")

    ground_truth = "maybe this moment works better cause some stuff on the page doesnt work as well as it does in real life"

    print(sr.forward(ground_truth, audio, fs, -480))

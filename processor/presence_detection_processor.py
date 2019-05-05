import gin
import numpy as np
from presence_detection.speech_rec import SpeechRec
from presence_detection.fb import PresenceScore
import logging

@gin.configurable
class PresenceDetectionProcessor:

    def __init__(self, threshold):
        gin.parse_config_file("presence_detection/config/presence_detection.gin")
        self.speech_rec_model = SpeechRec()
        self.presence_model = PresenceScore()
        self.logger = logging.getLogger('presenceDetectionProcessor')
        self.threshold = threshold

    def forward(self, ground_truth, audio, fs):
        log_probs = self.speech_rec_model.forward(audio, fs, display_chars=False)
        chars = self.speech_rec_model.alphabet._label_to_str + ["-"]
        score = self.presence_model.forward(ground_truth, log_probs, chars)
        self.logger.info("Presence detection of gt {} is {}".format(ground_truth, score))
        return score > self.threshold

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

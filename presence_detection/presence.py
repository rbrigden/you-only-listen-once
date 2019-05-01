import gin
import numpy as np
from presence_detection.speech_rec import SpeechRec
from presence_detection.fb import PresenceScore

class PresenceDetector:

    def __init__(self):
        self.speech_rec_model = SpeechRec()
        self.presence_model = PresenceScore()
        gin.parse_config_file("presence_detection/config/presence_detection.gin")

    def forward(self, audio, ):
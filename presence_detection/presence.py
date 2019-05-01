import gin
import numpy as np
from presence_detection.speech_rec import SpeechRec

class PresenceDetector:

    def __init__(self):
        self.speech_rec_model = SpeechRec()
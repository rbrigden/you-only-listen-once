
class SpeakerEmbeddingProcessor:

    def __init__(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
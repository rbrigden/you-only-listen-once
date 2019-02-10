import torch.nn as nn

class Verifier(nn.Module):

    def __init__(self, classifier):
        super(Verifier, self).__init__()
        self.embedding_size = classifier.embedding_size
        self.embed = lambda u: classifier.forward(u, em=True)

    def forward(self, utterances):
        u0, u1 = utterances
        e0 = self.embed(u0)
        e1 = self.embed(u1)





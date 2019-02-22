import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock


class Flatten(nn.Module):
    def forward(self, x):
        out = x.view(x.size()[0], -1)
        return out


class AvgPool(nn.Module):

    def __init__(self, dim):
        super(AvgPool, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, self.dim, keepdim=True)


class IdentifyAndEmbed(nn.Module):

    def __init__(self, nspeakers):
        super(IdentifyAndEmbed, self).__init__()
        self.relu = nn.ReLU()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=(2, 2), bias=False),
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.Conv2d(4, 16, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            BasicBlock(16, 16),
            nn.Conv2d(16, 64, kernel_size=3, stride=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            BasicBlock(64, 64),
            nn.Conv2d(64, 256, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=(2, 2), bias=True),
        )

        self.pool = nn.Sequential(AvgPool(2), Flatten())
        self.embedding_size = 512
        self.embedding = nn.Linear(896, self.embedding_size)
        self.ln = nn.LayerNorm(self.embedding_size)
        self.classification = nn.Linear(self.embedding_size, nspeakers)

    def _make_mask(self, utterance_shape, features_shape, seq_lens):
        _, _, ut, _ = utterance_shape
        _, _, ft, _ = features_shape
        scale = ut // ft
        mask = torch.zeros(features_shape).cuda()
        for i, l in enumerate(seq_lens):
            new_length = l // scale
            mask[i, :, :new_length, :] = 1
        return mask

    def forward(self, xs, em=False):
        """ Set em to skip the classification"""
        x, seq_lens = xs
        out = self.net(x)
        mask = self._make_mask(x.shape, out.shape, seq_lens)
        out *= mask
        out = self.pool(out)
        e = self.embedding(out)
        e = self.ln(F.relu(e))
        if em:
            return e
        c = self.classification(e)
        return F.log_softmax(c, dim=1), e



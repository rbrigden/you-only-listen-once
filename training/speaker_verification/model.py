import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
import  torchvision.models

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
            nn.Conv2d(1, 16, kernel_size=7, stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            BasicBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            BasicBlock(64, 64),
            nn.Conv2d(64, 256, kernel_size=3, stride=(1, 2), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=(2, 2), bias=True),
        )

        self.pool = nn.Sequential(AvgPool(2), Flatten())
        self.embedding_size = 256
        self.embedding = nn.Linear(768, self.embedding_size)
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
        # mask = self._make_mask(x.shape, out.shape, seq_lens)
        # out *= mask
        out = self.pool(out)
        e = self.embedding(out)
        e = self.ln(F.relu(e))
        if em:
            return e
        c = self.classification(e)
        return F.log_softmax(c, dim=1), e


class DenseNetWithEmbeddings(torchvision.models.DenseNet):

    def __init__(self, embedding_size, **kwargs):
        super(DenseNetWithEmbeddings, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.classifier = nn.Linear(embedding_size, self.classifier.out_features)
        self.embedding_layer = nn.Linear(1024, embedding_size)


    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        em = self.embedding_layer(out)
        logits = F.log_softmax(self.classifier(F.relu(em)), dim=1)
        return logits, em

def densnet121_with_embeddings(embedding_size, **kwargs):
    model = DenseNetWithEmbeddings(embedding_size, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


class DenseIdentifyAndEmbed(nn.Module):

    def __init__(self, nspeakers):
        super(DenseIdentifyAndEmbed, self).__init__()
        self.dense_net = densnet121_with_embeddings(256, num_classes=nspeakers)

    def forward(self, xs, em=False):
        x, seq_lens = xs
        out, embedding = self.dense_net(x)
        if em:
            return embedding
        return out, embedding



class RecurrentIdentifyAndEmbed(nn.Module):
    """ Bidirectional Pyramidal LSTM """

    def __init__(self, nspeakers, features=64):
        super(RecurrentIdentifyAndEmbed, self).__init__()
        hidden_size = 128
        self.rnns = nn.ModuleList([
            nn.LSTM(features, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True),
            nn.LSTM(hidden_size * 4, hidden_size=hidden_size * 2, num_layers=1, batch_first=True, bidirectional=True),
            nn.LSTM(hidden_size * 8, hidden_size=hidden_size * 4, num_layers=1, batch_first=True, bidirectional=True)
        ])

        self.embedding_size = 512
        self.embedding_layer = nn.Linear(8 * hidden_size, self.embedding_size)
        self.classification_layer = nn.Linear(self.embedding_size, nspeakers)
        self.ln = nn.LayerNorm(self.embedding_size)

    def _pyramid_forward(self, seqs, seq_lens):

        for idx, rnn in enumerate(self.rnns):
            packed_seqs = nn.utils.rnn.pack_padded_sequence(seqs, seq_lens, batch_first=True)
            out, (h, c) = rnn(packed_seqs)
            if idx < len(self.rnns) - 1:
                seqs, seq_lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
                if seqs.size(1) % 2 == 1:
                    seqs = seqs[:, :-1]
                b, t, f = seqs.size()
                nt, nf = (t // 2, f * 2)
                seqs = seqs.contiguous().view(b, nt, nf)
                seq_lens = seq_lens // 2

        return h.permute(1, 0, 2).contiguous().view(-1, 1024)


    def forward(self, xs, em=False):
        """ Set em to skip the classification"""
        x, seq_lens = xs
        out = self._pyramid_forward(x.squeeze(1), seq_lens)
        embedding = self.embedding_layer(out)
        if em:
            return embedding
        z = F.relu(embedding, inplace=True)
        z = self.ln(z)
        c = self.classification_layer(z)
        return F.log_softmax(c, dim=1), embedding







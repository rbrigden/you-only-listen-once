import torch
import torch.nn as nn
import torch.nn.functional as F
import  torchvision.models
from torchvision.models.resnet import BasicBlock
import gin

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



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


@gin.configurable
class IdentifyAndEmbed(nn.Module):

    def __init__(self, nspeakers):
        super(IdentifyAndEmbed, self).__init__()
        self.relu = nn.ReLU()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            BasicBlock5x5(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            BasicBlock5x5(64, 64),
            nn.Conv2d(64, 256, kernel_size=3, stride=(1, 2), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=(2, 2), bias=True),
        )

        self.pool = nn.Sequential(AvgPool(2), Flatten())
        self.embedding_size = 256
        self.embedding = nn.Linear(768, self.embedding_size)
        # self.ln = nn.LayerNorm(self.embedding_size)
        self.classification = nn.Linear(self.embedding_size, nspeakers)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

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
        out = self.pool(out)
        z = self.embedding(out)
        if em:
            return z
        z = 16 * nn.functional.normalize(z, p=2, dim=1)
        c = self.classification(z)
        return F.log_softmax(c, dim=1), z




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

        for name, param in self.rnns.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

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





class SimpleRecurrentEmbedAndIdentify(nn.Module):

    def __init__(self, nspeakers, features=64):
        super(SimpleRecurrentEmbedAndIdentify, self).__init__()
        hidden_size = 128
        self.rnn = nn.LSTM(features, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True),

        self.embedding_size = 128
        self.embedding_layer = nn.Linear(8 * hidden_size, self.embedding_size)
        self.classification_layer = nn.Linear(self.embedding_size, nspeakers)
        self.ln = nn.LayerNorm(self.embedding_size)

        for name, param in self.rnns.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

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


def conv5x5_(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels, out_channels, 5, padding=0, stride=stride, bias=False)


def conv3x3_(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1, bias=False)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = conv3x3_(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3_(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        return self.act(residual + self.bn2(self.conv2(out)))


class Group(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None, pool=False):
        super(Group, self).__init__()
        self.conv = conv5x5_(in_channels, out_channels, stride=stride)
        self.block1 = Block(out_channels, out_channels)
        self.act = nn.ELU(inplace=True)
        self.pool = pool

    def forward(self, x):
        x = self.act(self.conv(x))
        if self.pool:
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        return self.block1(x)


class ResNet(nn.Module):
    def __init__(self, nspeakers):
        super(ResNet, self).__init__()
        self.group1 = Group(1, 32, stride=2)
        self.group2 = Group(32, 64, stride=2)
        self.group3 = Group(64, 128, stride=2)
        self.group4 = Group(128, 256, stride=2)
        self.printed = False

        self.classifier = nn.Linear(256, nspeakers)
        self.act = nn.ELU(inplace=True)

        self.embedding_size = 256

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, em=False):
        embed = em
        x, seq_lens = x
        bsize = x.size(0)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)

        if not self.printed:
            print(x.shape)
            self.printed = True

        x = x.mean(dim=3).mean(dim=2).view(bsize, -1)  # Temporal Avg-Pool

        embeddings = x

        if embed:
            return embeddings
        else:
            embeddings = 16 * nn.functional.normalize(embeddings, p=2, dim=1)  # Scaled length-normalization
            logits = self.classifier(embeddings)
            return F.log_softmax(logits, dim=1), embeddings
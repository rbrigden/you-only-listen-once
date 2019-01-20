import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import _DenseBlock, _DenseLayer, _Transition
from torchvision.models.resnet import BasicBlock
from collections import OrderedDict

def conv1d(in_channels, out_channels, bias=True):
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)


def conv2d(in_channels, out_channels, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=bias)


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


class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock1d, self).__init__()
        self.conv1 = conv1d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv1d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock2d, self).__init__()
        self.conv1 = conv2d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv2d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class SpeakerClassifier1d(nn.Module):
    def __init__(self, features, outputs, hidden, nblocks):
        super(SpeakerClassifier1d, self).__init__()
        self.in_features = features
        self.block0 = nn.Sequential(conv1d(features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([ResidualBlock1d(hidden, hidden) for _ in range(nblocks)])
        self.embed_layer = nn.Linear(hidden, hidden)
        self.classification = nn.Linear(hidden, outputs)
        self.embedding_size = hidden
        self.relu = nn.ReLU(inplace=True)

    def embed(self, input_):
        x = input_
        x = self.block0(x)
        for block in self.blocks:
            x = block(x)
        avg_pooled = torch.mean(x, dim=2)
        embedding = self.embed_layer(self.relu(avg_pooled))
        return embedding

    def forward(self, input_):
        embed = F.relu(self.embed(input_))
        return self.classification(embed)



class SpeakerClassifierResidual2d(nn.Module):
    def __init__(self):
        # (4, 16, 64, 256, 128)
        super(SpeakerClassifierResidual2d, self).__init__()
        self.blocks = nn.Sequential(
            conv1d(1, 4),
            nn.BatchNorm2d(4),
            nn.ELU(),
            ResidualBlock2d(4, 16),
            ResidualBlock2d(16, 64)

        )

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 12, 6),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        # Linear layers 576
        self.embedding = nn.Linear(num_features, num_features)
        self.embedding_size = num_features
        self.classifier = nn.Linear(num_features, num_classes)
        self.bn1 = nn.BatchNorm1d(num_features)
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def embed(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = (out.sum(3).sum(2) / 62).view(-1, self.embedding_size)
        out = self.embedding(out)
        out = self.bn1 (out)
        out = F.relu(out, inplace=True)
        return out

    def forward(self, x, embed=False):
        out = self.embed(x)
        if embed:
            return out
        out = self.classifier(out)
        return out

class SpeakerClassifierDense(nn.Module):

    def __init__(self, nspeakers):
        super(SpeakerClassifierDense, self).__init__()
        # densnet 121
        self.network = DenseNet(num_init_features=64, growth_rate=8,
                block_config=(3, 3, 3, 3), num_classes=nspeakers)
        self.embedding_size = self.network.embedding_size

    def forward(self, x, embed=False):
        return F.log_softmax(self.network.forward(x, embed=embed), dim=1)


    def embed(self, x):
        return self.network.embed(x)




class SpeakerClassifier2d(nn.Module):

    def __init__(self, nspeakers):
        super(SpeakerClassifier2d, self).__init__()
        self.relu = nn.ReLU()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=(2, 2), bias=False),
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.Conv2d(4, 16, kernel_size=3, stride=(2, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=(2, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=(2, 2), bias=True),
            AvgPool(2),
            Flatten()
        )
        self.embedding_size = 512
        self.embedding = nn.Linear(896, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.classification = nn.Linear(512, nspeakers)

    def embed(self, x):
        out = self.net(x)
        out = self.embedding(out)
        out = self.bn1(out)
        out = self.relu(out)
        return out

    def forward(self, x, embed=False):
        embedding = self.embed(x)
        if embed:
            return embedding
        out = F.log_softmax(self.classification(embedding), dim=1)
        return out



class SpeakerEmbedding2d(nn.Module):


    def __init__(self, nspeakers):
        super(SpeakerEmbedding2d, self).__init__()
        self.relu = nn.ReLU()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=(2, 2), bias=False),
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.Conv2d(4, 16, kernel_size=3, stride=(2, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            BasicBlock(16, 16),
            nn.Conv2d(16, 64, kernel_size=3, stride=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            BasicBlock(64, 64),
            nn.Conv2d(64, 256, kernel_size=3, stride=(2, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=(2, 2), bias=True),
            AvgPool(2),
            Flatten()
        )
        self.embedding_size = 512
        self.embedding = nn.Linear(896, 512)
        self.ln = nn.LayerNorm(512)
        self.classification = nn.Linear(512, nspeakers)


    def forward(self, x, em=False):
        out = self.net(x)
        e = self.embedding(out)
        e = F.relu(self.ln(e))
        if em:
            return e
        c = self.classification(e)
        return F.log_softmax(c, dim=1), e

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


if __name__ == "__main__":
    model = SpeakerEmbedding2d(127)
    data = torch.randn(100, 1000, 64)
    out1, out2 = model(data.unsqueeze(1))
    print(out1.size())


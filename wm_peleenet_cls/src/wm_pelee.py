"""WM-PeleeNet reconstruction for 38-way wafer-map classification.

The paper specifies the block layout but not every channel width or CBAM
hyperparameter.  The choices below are documented reconstruction choices.
"""
import torch
from torch import nn


def conv(in_channels, out_channels, kernel_size, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = conv(3, 32, 3, 2)
        self.branch = nn.Sequential(conv(32, 16, 1), conv(16, 32, 3, 2))
        self.pool = nn.MaxPool2d(2, 2)
        self.fuse = conv(64, 32, 1)

    def forward(self, x):
        x = self.first(x)
        return self.fuse(torch.cat((self.branch(x), self.pool(x)), dim=1))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth):
        super().__init__()
        branch = growth // 2
        self.short = nn.Sequential(conv(in_channels, branch, 1), conv(branch, branch, 3))
        self.long = nn.Sequential(conv(in_channels, branch, 1), conv(branch, branch, 3), conv(branch, branch, 3))

    def forward(self, x):
        return torch.cat((x, self.short(x), self.long(x)), dim=1)


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        channel_weight = torch.sigmoid(self.channel(x.mean((2, 3), keepdim=True)) +
                                       self.channel(x.amax((2, 3), keepdim=True)))
        x = x * channel_weight
        spatial_weight = torch.sigmoid(self.spatial(torch.cat((x.mean(1, keepdim=True),
                                                              x.amax(1, keepdim=True)), dim=1)))
        return x * spatial_weight


class WMPeleeNet(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.stem = Stem()
        self.dense1 = nn.Sequential(DenseLayer(32, 48), DenseLayer(80, 48))
        self.transition1 = nn.Sequential(conv(128, 128, 1), nn.AvgPool2d(2, 2))
        self.dense2 = nn.Sequential(DenseLayer(128, 64), DenseLayer(192, 64))
        self.transition2 = conv(256, 256, 1)
        self.attention = CBAM(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.transition1(self.dense1(self.stem(x)))
        x = self.attention(self.transition2(self.dense2(x)))
        return self.classifier(self.pool(x).flatten(1))

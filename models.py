import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd


class QuartzNetDecoder(nn.Module):

    def __init__(self):

        super(QuartzNetDecoder, self).__init__()

        self.in_channels = 768
        self.out_channels = 80
        self.channels = [256, 256, 256, 512]
        self.k = [33, 39, 51, 1]
        self.n_blocks = 3

        self.conv1 = nn.Sequential(nn.Conv1d(self.in_channels, self.channels[0], kernel_size=self.k[0], padding=self.k[0]//2),
                                   nn.BatchNorm1d(self.channels[0], eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2, inplace=False))

        self.blocks = nn.ModuleList([])
        for i in range(self.n_blocks):
            pad = self.k[i] // 2
            self.blocks.append(JasperBlock(self.channels[i], self.channels[i+1], self.k[i], pad))

        self.conv2 = nn.Conv1d(self.channels[-1], self.out_channels, kernel_size=self.k[-1], padding=self.k[-1]//2)

    def forward(self, x):

        x = self.conv1(x)
        for layer in self.blocks:
            x = layer(x)

        return self.conv2(x)


class JasperBlock(nn.Module):

    def __init__(self, in_channels, out_channels, k, padding):

        super(JasperBlock, self).__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, out_channels, k, padding),
            ConvBlock(out_channels, out_channels, k, padding),
            ConvBlock(out_channels, out_channels, k, padding),
            ConvBlock(out_channels, out_channels, k, padding))

        self.last = nn.ModuleList([
            nn.Conv1d(out_channels, out_channels, kernel_size=k, stride=[1], padding=(padding,), dilation=[1], groups=out_channels, bias=False),
            nn.Conv1d(out_channels, out_channels, kernel_size=(1,), stride=(1,), bias=False),
            nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        ])

        self.res = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,), stride=[1], bias=False),
                                 nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x):

        y = self.res(x)
        x = self.blocks(x)

        for idx, layer in enumerate(self.last):
            x = layer(x)
            if idx == 2:
                x += y
                x = self.relu(x)
                x = self.dropout(x)

        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, k, padding):

        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=k, stride=[1], padding=(padding,), dilation=[1], groups=in_channels, bias=False),
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), stride=(1,), bias=False),
            nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False)
        )

    def forward(self, x):

        return self.layers(x)


class QuartzNetASR(nn.Module):

    def __init__(self):

        super(QuartzNetASR, self).__init__()

        self.in_channels = 80
        self.channels = [256, 256, 256, 512, 512, 512, 512, 1024, 29]
        self.k = [33, 39, 51, 63, 75, 87, 1, 1]
        self.n_blocks = 5

        self.conv1 = nn.Sequential(nn.Conv1d(self.in_channels, self.channels[0], kernel_size=self.k[0], stride=[1], padding=self.k[0]//2),
                                   nn.BatchNorm1d(self.channels[0], eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2, inplace=False))

        self.blocks = nn.ModuleList([])
        for i in range(self.n_blocks):
            pad = self.k[i] // 2
            self.blocks.append(JasperBlock(self.channels[i], self.channels[i+1], self.k[i], pad))

        self.conv2 = nn.Sequential(nn.Conv1d(self.channels[5], self.channels[6], kernel_size=87, padding=86, dilation=2),
                                   nn.BatchNorm1d(self.channels[6], eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2, inplace=False))
        self.conv3 = nn.Sequential(nn.Conv1d(self.channels[6], self.channels[7], kernel_size=1),
                                   nn.BatchNorm1d(self.channels[7], eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2, inplace=False))
        self.conv4 = nn.Conv1d(self.channels[7], self.channels[8], kernel_size=1)

    def forward(self, x):

        x = self.conv1(x)

        for layer in self.blocks:
            x = layer(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


class QuartzNetEncoder(nn.Module):

    def __init__(self):

        super(QuartzNetEncoder, self).__init__()

        self.in_channels = 80
        self.channels = [256, 256, 256, 512, 512]
        self.k = [33, 39, 51, 63]
        self.n_blocks = 4

        self.conv1 = nn.Sequential(nn.Conv1d(self.in_channels, self.channels[0], kernel_size=self.k[0], stride=[1], padding=self.k[0]//2),
                                   nn.BatchNorm1d(self.channels[0], eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU())

        self.blocks = nn.ModuleList([])
        for i in range(self.n_blocks):
            pad = self.k[i] // 2
            self.blocks.append(JasperBlock(self.channels[i], self.channels[i+1], self.k[i], pad))

    def forward(self, x):

        x = self.conv1(x)

        for layer in self.blocks:
            x = layer(x)

        return x

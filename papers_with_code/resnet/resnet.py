'''
Reference
paper link: https://arxiv.org/abs/1512.03385
code link: https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html
'''

import torch
import torch.nn as nn

def conv3x3(
    in_channels,
    out_channels,
    stride=1,
    padding=1
):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False
    )

def conv1x1(
    in_channels,
    out_channels,
    stride=1
):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=1, 
        stride=stride, 
        bias=False
    )

class BasicBlock(nn.Module):
    EXPANSION = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        norm_layer=nn.BatchNorm2d
    ):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)

        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = norm_layer(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # shortcut/skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out
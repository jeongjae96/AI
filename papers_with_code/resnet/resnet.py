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


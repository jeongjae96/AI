'''
Reference
- paper link: https://arxiv.org/abs/1512.03385
- code link: https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html
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
        projection=None,
        norm_layer=nn.BatchNorm2d
    ):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.projection = projection

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # shortcut/skip connection
        if self.projection is not None:
            identity = self.projection(x)
        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    EXPANSION = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        projection=None,
        norm_layer=nn.BatchNorm2d   
    ):
        super().__init__()

        '''
        Original implementation places the stride at the first 1x1 convolution(self.conv1).
        Placing the stride for downsampling at 3x3 convolution(self.conv2) improves accuracy.
        '''
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)

        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = norm_layer(out_channels)

        self.conv3 = conv1x1(out_channels, out_channels * self.EXPANSION)
        self.bn3 = norm_layer(out_channels * self.EXPANSION)

        self.relu = nn.ReLU(inplace=True)

        self.projection = projection

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.projection is not None:
            identity = self.projection(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        norm_layer=nn.BatchNorm2d
    ):
        super().__init__()

        self._norm_layer = norm_layer
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3, 
            self.in_channels, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        self.bn1 = norm_layer(self.in_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 258, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.EXPANSION, num_classes)

    def _make_layer(
        self,
        block,
        channels,
        blocks,
        stride=1,
    ):
        norm_layer = self._norm_layer
        projection = None

        if stride != 1 or self.in_channels != channels * block.EXPANSION:
            projection = nn.Sequential(
                conv1x1(self.in_channels, channels * block.EXPANSION, stride),
                norm_layer(channels * block.EXPANSION)
            )

        layers = []
        
        layers.append(
            block(
                self.in_channels,
                channels,
                stride,
                projection,
                norm_layer
            )
        )

        self.in_channels = channels * block.EXPANSION

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
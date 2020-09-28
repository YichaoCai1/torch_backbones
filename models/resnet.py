# -*- coding: UTF-8 -*-
"""
An unofficial implementation of resnet with pytorch
@Cai Yichao 2020_09_08
"""

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.blocks.residual_blocks import BasicBlock, BottleNeck


class ResNet(nn.Module):
    """
    building ResNet_34
    """

    def __init__(self, block: object, groups: object, num_classes, is_se=False) -> object:
        super(ResNet, self).__init__()
        self.channels = 64  # out channels from the first convolutional layer
        self.block = block
        self.is_se = is_se

        self.conv1 = nn.Conv2d(3, self.channels, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.channels)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2_x = self._make_conv_x(channels=64, blocks=groups[0], strides=1, index=2)
        self.conv3_x = self._make_conv_x(channels=128, blocks=groups[1], strides=2, index=3)
        self.conv4_x = self._make_conv_x(channels=256, blocks=groups[2], strides=2, index=4)
        self.conv5_x = self._make_conv_x(channels=512, blocks=groups[3], strides=2, index=5)
        self.pool2 = nn.AvgPool2d(7)
        patches = 512 if self.block.message == "basic" else 512 * 4
        self.fc = nn.Linear(patches, num_classes)  # for 224 * 224 input size

    def _make_conv_x(self, channels, blocks, strides, index):
        """
        making convolutional group
        :param channels: output channels of the conv-group
        :param blocks: number of blocks in the conv-group
        :param strides: strides
        :return: conv-group
        """
        list_strides = [strides] + [1] * (blocks - 1)  # In conv_x groups, the first strides is 2, the others are ones.
        conv_x = nn.Sequential()
        for i in range(len(list_strides)):
            layer_name = str("block_%d_%d" % (index, i))  # when use add_module, the name should be difference.
            conv_x.add_module(layer_name, self.block(self.channels, channels, list_strides[i], self.is_se))
            self.channels = channels if self.block.message == "basic" else channels * 4
        return conv_x

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn(out))
        out = self.pool1(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out))
        return out


def ResNet_18(num_classes=1000):
    return ResNet(block=BasicBlock, groups=[2, 2, 2, 2], num_classes=num_classes)


def ResNet_34(num_classes=1000):
    return ResNet(block=BasicBlock, groups=[3, 4, 6, 3], num_classes=num_classes)


def ResNet_50(num_classes=1000):
    return ResNet(block=BottleNeck, groups=[3, 4, 6, 3], num_classes=num_classes)


def ResNet_101(num_classes=1000):
    return ResNet(block=BottleNeck, groups=[3, 4, 23, 3], num_classes=num_classes)


def ResNet_152(num_classes=1000):
    return ResNet(block=BottleNeck, groups=[3, 8, 36, 3], num_classes=num_classes)


def ResNet_50_SE(num_classes=1000):
    return ResNet(block=BottleNeck, groups=[3, 4, 6, 3], num_classes=num_classes, is_se=True)

# def test():
#     # net = ResNet_18()
#     # net = ResNet_34()
#     # net = ResNet_50()
#     # net = ResNet_101()
#     # net = ResNet_152()
#     net = ResNet_50_SE()
#     # net = ResNet_50()
#     summary(net, (3, 224, 224))
#
#
# test()

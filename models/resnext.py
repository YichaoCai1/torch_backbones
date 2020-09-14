# -*- coding: UTF-8 -*-
"""
An unofficial implementation of ResNeXt with pytorch
@Cai Yichao 2020_09_14
"""
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from inception_blocks import BN_Conv2d

class ResNeXt_Block(nn.Module):
    """
    ResNeXt block with group convolutions
    """

    def __init__(self, in_chnls, out_chnls, cardinality, group_depth, stride):
        super(ResNeXt_Block, self).__init__()
        self.group_chnls = cardinality * group_depth
        self.conv1 = BN_Conv2d(in_chnls, self.group_chnls, 1, stride=1, padding=0)
        self.conv2 = BN_Conv2d(self.group_chnls, self.group_chnls, 3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.group_chnls, out_chnls, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_chnls)
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_chnls, out_chnls, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_chnls)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        out += self.short_cut(x)
        return F.relu(out)


class ResNeXt(nn.Module):
    """
    ResNeXt builder
    """

    def __init__(self, layers: object, cardinality, group_depth, num_classes) -> object:
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.group_depth = group_depth
        self.channels = 64
        self.conv1 = BN_Conv2d(3, self.channels, 7, stride=2, padding=3)
        self.conv2 = self.___make_layers(256, layers[0], stride=1)
        self.conv3 = self.___make_layers(512, layers[1], stride=2)
        self.conv4 = self.___make_layers(1024, layers[2], stride=2)
        self.conv5 = self.___make_layers(2048, layers[3], stride=2)
        self.fc = nn.Linear(2048, num_classes)   # 224x224 input size

    def ___make_layers(self, out_chnls, blocks, stride):
        strides = [stride] + [1] * (blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNeXt_Block(self.channels, out_chnls, self.cardinality, self.group_depth, stride))
            self.channels = out_chnls
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape, "--------1")
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        # print(out.shape, "--------2")
        out = self.conv3(out)
        # print(out.shape, "--------3")
        out = self.conv4(out)
        # print(out.shape, "--------4")
        out = self.conv5(out)
        # print(out.shape, "--------5")
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out))
        return out


def resNeXt29_8x64d(num_classes=1000):
    return ResNeXt([3, 3, 3, 3], 8, 64, num_classes)


def resNeXt29_16x64d(num_classes=1000):
    return ResNeXt([3, 3, 3, 3], 16, 64, num_classes)


def resNeXt50_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 6, 3], 32, 4, num_classes)


def resNeXt101_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 23, 3], 32, 4, num_classes)


def resNeXt101_64x4d(num_classes=1000):
    return ResNeXt([3, 4, 23, 3], 64, 4, num_classes)


def test():
    # net = resNeXt29_8x64d()
    # net = resNeXt29_16x64d()
    # net = resNeXt50_32x4d()
    # net = resNeXt101_32x4d()
    net = resNeXt101_64x4d()
    summary(net, (3, 224, 224))


test()





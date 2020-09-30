# -*- coding: UTF-8 -*-
"""
An unofficial implementation of Inception_v4,
Inception-ResNet-v1, Inception-ResNet-v2 with pytorch
@Cai Yichao 2020_09_011
"""

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.blocks.conv_bn import BN_Conv2d
from models.blocks.inception_blocks import *


class Inception(nn.Module):
    """
    implementation of Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2
    """

    def __init__(self, version, num_classes, is_se=False):
        super(Inception, self).__init__()
        self.version = version
        self.stem = Stem_Res1() if self.version == "res1" else Stem_v4_Res2()
        self.inception_A = self.__make_inception_A()
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()
        if self.version == "v4":
            self.fc = nn.Linear(1536, num_classes)
        elif self.version == "res1":
            self.fc = nn.Linear(1792, num_classes)
        else:
            self.fc = nn.Linear(2144, num_classes)

    def __make_inception_A(self):
        layers = []
        if self.version == "v4":
            for _ in range(4):
                layers.append(Inception_A(384, 96, 96, 64, 96, 64, 96))
        elif self.version == "res1":
            for _ in range(5):
                layers.append(Inception_A_res(256, 32, 32, 32, 32, 32, 32, 256))
        else:
            for _ in range(5):
                layers.append(Inception_A_res(384, 32, 32, 32, 32, 48, 64, 384))
        return nn.Sequential(*layers)

    def __make_reduction_A(self):
        if self.version == "v4":
            return Reduction_A(384, 192, 224, 256, 384)  # 1024
        elif self.version == "res1":
            return Reduction_A(256, 192, 192, 256, 384)  # 896
        else:
            return Reduction_A(384, 256, 256, 384, 384)  # 1152

    def __make_inception_B(self):
        layers = []
        if self.version == "v4":
            for _ in range(7):
                layers.append(Inception_B(1024, 128, 384, 192, 224, 256,
                                          192, 192, 224, 224, 256))  # 1024
        elif self.version == "res1":
            for _ in range(10):
                layers.append(Inception_B_res(896, 128, 128, 128, 128, 896))  # 896
        else:
            for _ in range(10):
                layers.append(Inception_B_res(1152, 192, 128, 160, 192, 1152))  # 1152
        return nn.Sequential(*layers)

    def __make_reduction_B(self):
        if self.version == "v4":
            return Reduction_B_v4(1024, 192, 192, 256, 256, 320, 320)  # 1536
        elif self.version == "res1":
            return Reduction_B_Res(896, 256, 384, 256, 256, 256, 256, 256)  # 1792
        else:
            return Reduction_B_Res(1152, 256, 384, 256, 288, 256, 288, 320)  # 2144

    def __make_inception_C(self):
        layers = []
        if self.version == "v4":
            for _ in range(3):
                layers.append(Inception_C(1536, 256, 256, 384, 256, 384, 448, 512, 256))
        elif self.version == "res1":
            for _ in range(5):
                layers.append(Inception_C_res(1792, 192, 192, 192, 192, 1792))
        else:
            for _ in range(5):
                layers.append(Inception_C_res(2144, 192, 192, 224, 256, 2144))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.inception_A(out)
        out = self.Reduction_A(out)
        out = self.inception_B(out)
        out = self.Reduction_B(out)
        out = self.inception_C(out)
        out = F.avg_pool2d(out, 8)
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.fc(out)
        return F.softmax(out)


def inception_v4(classes=1000):
    return Inception("v4", classes)


def inception_resnet_v1(classes=1000):
    return Inception("res1", classes)


def inception_resnet_v2(classes=1000):
    return Inception("res2", classes)

# def test():
#     net = inception_v4()
#     # net = inception_resnet_v1()
#     # net = inception_resnet_v2()
#     summary(net, (3, 299, 299))
#
# test()

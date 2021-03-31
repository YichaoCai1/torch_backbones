# -*- coding: UTF-8 -*-
"""
An unofficial implementation of ShuffleNet-v2 with pytorch
@Cai Yichao 2020_10_10
"""

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.blocks.shuffle_block import *


class ShuffleNet_v2(nn.Module):
    """ShuffleNet-v2"""

    _defaults = {
        "sets": {0.5, 1, 1.5, 2},
        "units": [3, 7, 3],
        "chnl_sets": {0.5: [24, 48, 96, 192, 1024],
                      1: [24, 116, 232, 464, 1024],
                      1.5: [24, 176, 352, 704, 1024],
                      2: [24, 244, 488, 976, 2048]}
    }

    def __init__(self, scale, num_cls, is_se=False, is_res=False) -> object:
        super(ShuffleNet_v2, self).__init__()
        self.__dict__.update(self._defaults)
        assert (scale in self.sets)
        self.is_se = is_se
        self.is_res = is_res
        self.chnls = self.chnl_sets[scale]

        # make layers
        self.conv1 = BN_Conv2d(3, self.chnls[0], 3, 2, 1)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.stage2 = self.__make_stage(self.chnls[0], self.chnls[1], self.units[0])
        self.stage3 = self.__make_stage(self.chnls[1], self.chnls[2], self.units[1])
        self.stage4 = self.__make_stage(self.chnls[2], self.chnls[3], self.units[2])
        self.conv5 = BN_Conv2d(self.chnls[3], self.chnls[4], 1, 1, 0)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.body = self.__make_body()
        self.fc = nn.Linear(self.chnls[4], num_cls)

    def __make_stage(self, in_chnls, out_chnls, units):
        layers = [DSampling(in_chnls),
                  BasicUnit(2 * in_chnls, out_chnls, self.is_se, self.is_res)]
        for _ in range(units-1):
            layers.append(BasicUnit(out_chnls, out_chnls, self.is_se, self.is_res))
        return nn.Sequential(*layers)

    def __make_body(self):
        return nn.Sequential(
            self.conv1, self.maxpool, self.stage2, self.stage3,
            self.stage4, self.conv5, self.globalpool
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.softmax(out)


"""
API
"""


def shufflenet_0_5x(num_classes=1000):
    return ShuffleNet_v2(0.5, num_classes)


def shufflenet_0_5x_se(num_classes=1000):
    return ShuffleNet_v2(0.5, num_classes, is_se=True)


def shufflenet_0_5x_res(num_classes=1000):
    return ShuffleNet_v2(0.5, num_classes, is_res=True)


def shufflenet_0_5x_se_res(num_classes=1000):
    return ShuffleNet_v2(0.5, num_classes, is_se=True, is_res=True)


def shufflenet_1x(num_classes=1000):
    return ShuffleNet_v2(1, num_classes)


def shufflenet_1x_se(num_classes=1000):
    return ShuffleNet_v2(1, num_classes, is_se=True)


def shufflenet_1x_res(num_classes=1000):
    return ShuffleNet_v2(1, num_classes, is_res=True)


def shufflenet_1x_se_res(num_classes=1000):
    return ShuffleNet_v2(1, num_classes, is_se=True, is_res=True)


def shufflenet_1_5x(num_classes=1000):
    return ShuffleNet_v2(1.5, num_classes)


def shufflenet_1_5x_se(num_classes=1000):
    return ShuffleNet_v2(1.5, num_classes, is_se=True)


def shufflenet_1_5x_res(num_classes=1000):
    return ShuffleNet_v2(1.5, num_classes, is_res=True)


def shufflenet_1_5x_se_res(num_classes=1000):
    return ShuffleNet_v2(1.5, num_classes, is_se=True, is_res=True)


def shufflenet_2x(num_classes=1000):
    return ShuffleNet_v2(2, num_classes)


def shufflenet_2x_se(num_classes=1000):
    return ShuffleNet_v2(2, num_classes, is_se=True)


def shufflenet_2x_res(num_classes=1000):
    return ShuffleNet_v2(2, num_classes, is_res=True)


def shufflenet_2x_se_res(num_classes=1000):
    return ShuffleNet_v2(2, num_classes, is_se=True, is_res=True)

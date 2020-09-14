# -*- coding: UTF-8 -*-
"""
An unofficial implementation of Inception_ResNet with pytorch
@Cai Yichao 2020_09_011
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from inception_blocks import BN_Conv2d\
    Inception_A_res, Inception_B_res, Inception_C_res, \
    Reduction_A, Reduction_B_v4, Stem_v4_Res2, Stem_Res1

class Inception_ResNet(nn.Module):
    """
    inception-resnet-v1, v2
    """

    def __init__(self, version, num_classes=1000):
        super(Inception_ResNet, self).__init__()


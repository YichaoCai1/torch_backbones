# -*- coding: UTF-8 -*-
"""
An unofficial implementation of Inception-v3 with pytorch
without Auxiliary classifier
@Cai Yichao 2020_09_09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.blocks.conv_bn import BN_Conv2d


class Block_bank(nn.Module):
    """
    inception structures
    """

    def __init__(self, block_type, in_channels, b1_reduce, b1, b2_reduce, b2, b3, b4):
        super(Block_bank, self).__init__()
        self.block_type = block_type  # controlled by strings "type1", "type2", "type3", "type4", "type5"

        """
        branch 1
        """
        # reduce, 3x3, 3x3
        self.branch1_type1 = nn.Sequential(
            BN_Conv2d(in_channels, b1_reduce, 1, 1, 0, bias=False),
            BN_Conv2d(b1_reduce, b1, 3, 1, 1, bias=False),
            BN_Conv2d(b1, b1, 3, 1, 1, bias=False)
        )

        # reduce, 3x3, 3x3_s2
        self.branch1_type2 = nn.Sequential(
            BN_Conv2d(in_channels, b1_reduce, 1, 1, 0, bias=False),
            BN_Conv2d(b1_reduce, b1, 3, 1, 1, bias=False),
            BN_Conv2d(b1, b1, 3, 2, 0, bias=False)
        )

        # reduce, 1x7, 7x1, 1x7, 7x1
        self.branch1_type3 = nn.Sequential(
            BN_Conv2d(in_channels, b1_reduce, 1, 1, 0, bias=False),
            BN_Conv2d(b1_reduce, b1_reduce, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(b1_reduce, b1_reduce, (1, 7), (1, 1), (0, 3), bias=False),  # same padding
            BN_Conv2d(b1_reduce, b1_reduce, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(b1_reduce, b1, (1, 7), (1, 1), (0, 3), bias=False)
        )

        # reduce, 1x7, 7x1, 3x3_s2
        self.branch1_type4 = nn.Sequential(
            BN_Conv2d(in_channels, b1_reduce, 1, 1, 0, bias=False),
            BN_Conv2d(b1_reduce, b1, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b1, b1, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(b1, b1, 3, 2, 0, bias=False)
        )

        # reduce, 3x3, 2 sub-branch of 1x3, 3x1
        self.branch1_type5_head = nn.Sequential(
            BN_Conv2d(in_channels, b1_reduce, 1, 1, 0, bias=False),
            BN_Conv2d(b1_reduce, b1, 3, 1, 1, bias=False)
        )
        self.branch1_type5_body1 = BN_Conv2d(b1, b1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch1_type5_body2 = BN_Conv2d(b1, b1, (3, 1), (1, 1), (1, 0), bias=False)

        """
        branch 2
        """
        # reduce, 5x5
        self.branch2_type1 = nn.Sequential(
            BN_Conv2d(in_channels, b2_reduce, 1, 1, 0, bias=False),
            BN_Conv2d(b2_reduce, b2, 5, 1, 2, bias=False)
        )

        # 3x3_s2
        self.branch2_type2 = BN_Conv2d(in_channels, b2, 3, 2, 0, bias=False)

        # reduce, 1x7, 7x1
        self.branch2_type3 = nn.Sequential(
            BN_Conv2d(in_channels, b2_reduce, 1, 1, 0, bias=False),
            BN_Conv2d(b2_reduce, b2_reduce, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b2_reduce, b2, (7, 1), (1, 1), (3, 0), bias=False)
        )

        # reduce, 3x3_s2
        self.branch2_type4 = nn.Sequential(
            BN_Conv2d(in_channels, b2_reduce, 1, 1, 0, bias=False),
            BN_Conv2d(b2_reduce, b2, 3, 2, 0, bias=False)
        )

        # reduce, 2 sub-branch of 1x3, 3x1
        self.branch2_type5_head = BN_Conv2d(in_channels, b2_reduce, 1, 1, 0, bias=False)
        self.branch2_type5_body1 = BN_Conv2d(b2_reduce, b2, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch2_type5_body2 = BN_Conv2d(b2_reduce, b2, (3, 1), (1, 1), (1, 0), bias=False)

        """
        branch 3
        """
        # avg pool, 1x1
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, b3, 1, 1, 0, bias=False)
        )

        """
        branch 4
        """
        # 1x1
        self.branch4 = BN_Conv2d(in_channels, b4, 1, 1, 0, bias=False)

    def forward(self, x):
        if self.block_type == "type1":
            out1 = self.branch1_type1(x)
            out2 = self.branch2_type1(x)
            out3 = self.branch3(x)
            out4 = self.branch4(x)
            out = torch.cat((out1, out2, out3, out4), 1)
        elif self.block_type == "type2":
            out1 = self.branch1_type2(x)
            out2 = self.branch2_type2(x)
            out3 = F.max_pool2d(x, 3, 2, 0)
            out = torch.cat((out1, out2, out3), 1)
        elif self.block_type == "type3":
            out1 = self.branch1_type3(x)
            out2 = self.branch2_type3(x)
            out3 = self.branch3(x)
            out4 = self.branch4(x)
            out = torch.cat((out1, out2, out3, out4), 1)
        elif self.block_type == "type4":
            out1 = self.branch1_type4(x)
            out2 = self.branch2_type4(x)
            out3 = F.max_pool2d(x, 3, 2, 0)
            out = torch.cat((out1, out2, out3), 1)
        else:  # type5
            tmp = self.branch1_type5_head(x)
            out1_1 = self.branch1_type5_body1(tmp)
            out1_2 = self.branch1_type5_body2(tmp)
            tmp = self.branch2_type5_head(x)
            out2_1 = self.branch2_type5_body1(tmp)
            out2_2 = self.branch2_type5_body2(tmp)
            out3 = self.branch3(x)
            out4 = self.branch4(x)
            out = torch.cat((out1_1, out1_2, out2_1, out2_2, out3, out4), 1)

        return out


class Inception_v3(nn.Module):
    def __init__(self, num_classes):
        super(Inception_v3, self).__init__()
        self.conv = BN_Conv2d(3, 32, 3, 2, 0, bias=False)
        self.conv1 = BN_Conv2d(32, 32, 3, 1, 0, bias=False)
        self.conv2 = BN_Conv2d(32, 64, 3, 1, 1, bias=False)
        self.conv3 = BN_Conv2d(64, 80, 1, 1, 0, bias=False)
        self.conv4 = BN_Conv2d(80, 192, 3, 1, 0, bias=False)
        self.inception1_1 = Block_bank("type1", 192, 64, 96, 48, 64, 32, 64)
        self.inception1_2 = Block_bank("type1", 256, 64, 96, 48, 64, 64, 64)
        self.inception1_3 = Block_bank("type1", 288, 64, 96, 48, 64, 64, 64)
        self.inception2 = Block_bank("type2", 288, 64, 96, 288, 384, 288, 288)
        self.inception3_1 = Block_bank("type3", 768, 128, 192, 128, 192, 192, 192)
        self.inception3_2 = Block_bank("type3", 768, 160, 192, 160, 192, 192, 192)
        self.inception3_3 = Block_bank("type3", 768, 160, 192, 160, 192, 192, 192)
        self.inception3_4 = Block_bank("type3", 768, 192, 192, 192, 192, 192, 192)
        self.inception4 = Block_bank("type4", 768, 192, 192, 192, 320, 288, 288)
        self.inception5_1 = Block_bank("type5", 1280, 448, 384, 384, 384, 192, 320)
        self.inception5_2 = Block_bank("type5", 2048, 448, 384, 384, 384, 192, 320)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = F.max_pool2d(out, 3, 2, 0)
        out = self.conv3(out)
        out = self.conv4(out)
        out = F.max_pool2d(out, 3, 2, 0)
        out = self.inception1_1(out)
        out = self.inception1_2(out)
        out = self.inception1_3(out)
        out = self.inception2(out)
        out = self.inception3_1(out)
        out = self.inception3_2(out)
        out = self.inception3_3(out)
        out = self.inception3_4(out)
        out = self.inception4(out)
        out = self.inception5_1(out)
        out = self.inception5_2(out)
        out = F.avg_pool2d(out, 8)
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # return F.softmax(out)
        return out


def inception_v3(num_classes=1000):
    return Inception_v3(num_classes)

# def test():
#     net = inception_v3()
#     summary(net, (3, 299, 299))
#
#
# test()

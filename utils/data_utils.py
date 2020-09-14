# -*- coding:UTF-8 -*-
"""
data loader
@Cai Yichao 2020_09_08
"""

import torch
import torchvision
import torchvision.transforms as transforms


def gen_train_loader(path, input_size, batch_size):
    train_set = torchvision.datasets.ImageFolder(path, transform=transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomCrop(input_size, padding=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return loader


def gen_test_loader(path, input_size, batch_size):
    test_set = torchvision.datasets.ImageFolder(path, transform=transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))
    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return loader


# -*- coding:UTF-8 -*-
"""
tesing my backbone implementations
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchsummary import summary

from models.resnet import *
from utils.arg_utils import *
from utils.data_utils import *
from utils.progress_utils import progress_bar

"""
loading args
"""
args = fetch_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
loading data-set....
"""
print("==> loading data-set...")
train_loader = gen_train_loader(args['train_path'], args['input_size'], args['train_batch_size'])
test_loader = gen_test_loader(args['test_path'], args['input_size'], args['test_batch_size'])
classes = args['classes']
print('Task classes are: ', classes)

"""
model
"""
print("==> building model...")
net = ResNet_50(num_classes=2)
summary(net, (3, 224, 224))

net = net.to(device)
if device is 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

"""
training
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args['learning_rate'], momentum=0.9, weight_decay=5e-4)


def train(epoch):
    print('\nEpoch:[%d/%d]' % (epoch, args['epochs']))
    net.train()
    loss, correct, total = 0, 0, 0

    for index, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss_curr = criterion(outputs, targets)
        loss_curr.backward()
        optimizer.step()

        loss += loss_curr.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(index, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (loss / (index + 1), 100. * correct / total, correct, total))


best_acc = 0
def test(epoch):
    global best_acc
    net.eval()
    loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss_curr = criterion(outputs, targets)

            loss += loss_curr.item()
            _, predicted = outputs.max(1)
            print(targets)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


            progress_bar(index, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         %(loss/(index+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print("Saving checkpoints..")
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args['ckpt_path']):
            os.mkdir(args['ckpt_path'])
        torch.save(state, str('./checkpoint/ckpt_%d_acc%.2f.t7' % (epoch, acc)))
        best_acc = acc


for epoch in range(args['epochs']):
    train(epoch)
    test(epoch)
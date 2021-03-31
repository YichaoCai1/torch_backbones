# -*- coding:UTF-8 -*-
"""
training classifying task with CNN
@Cai Yichao 2020_09_18
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchsummary import summary

from models.resnet import *
from models.resnext import *
from models.densenet import *
from utils.arg_utils import *
from utils.data_utils import *
from utils.progress_utils import progress_bar
from utils.earlystopping import EarlyStopping

"""
arguments
"""
args = fetch_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
early_stopping = EarlyStopping(args['patience'], verbose=True, delta=args['delta'])


"""
loading data-set....
"""
print("==> loading data-set...")
train_loader, classes = gen_train_loader(args['train_path'], args['input_size'], args['train_batch_size'])
test_loader, _ = gen_test_loader(args['test_path'], args['input_size'], args['test_batch_size'])
print('Task classes are: ', classes)
num_classes = len(classes)
print(num_classes)


"""
model
"""
print("==> building model...")
# net = ResNet_50(num_classes=2)
# net = resNeXt50_32x4d_SE(num_classes=num_classes)
net = densenet_121(num_classes=num_classes)
net = net.to(device)
summary(net, (3, 224, 224))

# if device is 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True


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

    eval_loss = loss/(index+1)
    acc = 100.*correct/total
    if acc >= best_acc:
        print("Saving checkpoints..")
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'eval_loss': eval_loss
        }
        if not os.path.isdir(args['ckpt_path']):
            os.mkdir(args['ckpt_path'])
        torch.save(state, args['ckpt_path'] + str('/ckpt_%d_acc%.2f.pt' % (epoch, acc)))
        best_acc = acc

    return eval_loss


for epoch in range(args['epochs']):
    train(epoch)
    eval_loss = test(epoch)

    # early stopping
    early_stopping(eval_loss, net)
    if early_stopping.early_stop:
        print("Early stopping.")
        break

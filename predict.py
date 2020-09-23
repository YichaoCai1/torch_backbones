# -*- coding:UTF-8 -*-
"""
predict
@Cai Yichao 2020_09_23
"""

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from models.resnext import *
from utils.arg_utils import *
from PIL import Image
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str)
parse_args = parser.parse_args()

args = fetch_args()
classes = ['infrared', 'normal']
num_classes = len(classes)
ckpt_file = args['ckpt_path']+'/ckpt_3_acc100.00.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.Resize(args['input_size']),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
"""
loading model
"""
net = resNeXt50_32x4d_SE(num_classes=num_classes)
ckpt = torch.load(ckpt_file)
# if device is 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

net.to(device)
net.load_state_dict(ckpt['net'])
net.eval()

start_time = time.time()
image = Image.open(parse_args.file)
image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

with torch.no_grad():
    out = net(image)
    _, predict = out.max(1)
    print(classes[predict[0]])
print("cost time: %.2f"%(time.time()-start_time))

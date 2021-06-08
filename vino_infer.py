import sys
import os
import numpy as np
from PIL import Image
import time
import argparse
import torchvision.transforms as transforms
from openvino.inference_engine import IECore

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str)
parse_args = parser.parse_args()

classes = ['infrared', 'normal']

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

ie = IECore()
net = ie.read_network(model="IR_ckpt/model.xml")
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))
n, c, h, w = net.input_info[input_blob].input_data.shape
exec_net = ie.load_network(network=net, device_name='CPU')

start_time = time.time()
image = Image.open(parse_args.file)
image = transform(image)
image = image.unsqueeze(0).cpu().numpy()

res = exec_net.infer(inputs={input_blob: image})
res = res[out_blob]
predict = np.argmax(res, axis=1)
print(classes[predict[0]])
print("vino cost time: %.4f"%(time.time()-start_time))
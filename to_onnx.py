import torch
from models.shufflenet_v2 import *
import onnx

# load model
net = shufflenet_1x_se(num_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load("toch_ckpt/model.pt", map_location=device)
net.load_state_dict(ckpt["net"])
net.to(device)
net.eval()

# 转onnx
dummy_input = torch.randn(1, 3, 224, 224)
input_names = ["input"]
output_names = ['output']
torch.onnx.export(net, dummy_input, 'onnx_ckpt/model.onnx', verbose=True,\
     input_names=input_names, output_names=output_names)
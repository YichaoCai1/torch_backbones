from torchsummary import summary
from torchstat import stat
from models.densenet import *
from models.darknet import *
from models.resnet import *
from models.csp_resnext import csp_resnext_50_32x4d
from models.csp_darknet import csp_darknet_53
from models.mnasnet import MnasNet_A1

# net = ResNet_50()
# summary(net, (3, 224, 224))

net = ResNet_50()
summary(net, (3, 224, 224))

# net = densenet_201(num_classes=1000)
# stat(net, (3, 224, 224))
#
# net = csp_densenet_201(num_classes=1000)
# stat(net, (3, 224, 224))

# net = darknet_53()
# summary(net, (3, 224, 224))

# net = csp_resnext_50_32x4d()
# summary(net, (3, 256, 256))

# net = csp_darknet_53()
# summary(net, (3, 256, 256))

# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
#
# x = torch.linspace(-5, 5, 200)
# x = Variable(x)
# x_np = x.data.numpy()
#
# y_relu = F.relu(x).data.numpy()
# y_softplus = F.softplus(x).data.numpy()
# y_mish = x * F.tanh(F.softplus(x)).data.numpy()
#
# plt.figure(1, figsize=(8, 6))
# # plt.plot(x_np, y_relu, c='blue', label='ReLU')
# # plt.plot(x_np, y_softplus, c='red', label='softplus')
# plt.plot(x_np, y_mish, c='red', label='Mish')
# plt.legend(loc='best')
#
# plt.show()

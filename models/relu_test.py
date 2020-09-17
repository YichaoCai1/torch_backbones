
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

test = np.array([[[[-0.6]], [[6.5]], [[-0.44]], [[7.35]], [[0.27]]]])
test_t = torch.from_numpy(test)
print(test_t.shape)
out = F.relu(test_t)
print(out)
out = F.sigmoid(test_t)
print(out)
import time

import torch
import matplotlib.pyplot as plt



x = time.time()

# a = torch.randn((10,60,480,640)).cuda()
a = torch.randn((10,60,480,640),device='cuda')


y = time.time()
print(y-x)

import matplotlib.pyplot as plt
import numpy as np
import torch
import PIL
from PIL import Image
import time
indexes = torch.arange(0,100000, 1).float()
a=time.time()
for i in range(100):
    selected_indexes = torch.multinomial(indexes, 1500, replacement=True)
# selected_indexes = torch.multinomial(indexes, 500, replacement=True)
# selected_indexes = torch.multinomial(indexes, 500, replacement=True)
b=time.time()
print(b-a)
print(selected_indexes)
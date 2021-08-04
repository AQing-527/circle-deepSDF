import os
import numpy as np
import cv2
import torch
from net import SDFNet

PTH_DATA_PATH = "../models/"


print('Enter shape name:')
name = input()

data = torch.load(f'{PTH_DATA_PATH}{name}.pth')

# weights = data["model"]

for k, v in data.items():
    print(k)
    print(v.shape)


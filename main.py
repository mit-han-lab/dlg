# -*- coding: utf-8 -*-
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

dst = datasets.CIFAR100("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = 25
gt_data = tp(dst[img_index][0]).to(device)
gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )

plt.imshow(tt(gt_data[0].cpu()))

from model.vision import LeNet
net = LeNet().to(device)
    
net.apply(weights_init)
criterion = nn.CrossEntropyLoss()

# compute original gradient 
out = net(gt_data)
y = criterion(out, gt_label)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))
original_dy_dx_fp16 = list((_.detach().clone().half().float() for _ in dy_dx))

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
plt.imshow(tt(dummy_data[0].cpu()))

optimizer = torch.optim.LBFGS([dummy_data, ] )

condition = True
prev_loss = 0
counter = 0 

history = []
for iters in range(300):
    def closure():
        optimizer.zero_grad()

        pred = net(dummy_data) 
        dummy_loss = criterion(pred, gt_label) 
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        
        return grad_diff
    
    optimizer.step(closure)
    if iters % 10 == 0: 
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')


# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot
import random
from torch.distributions.laplace import Laplace

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str,default="",
                    help='the path to customized image.')
args = parser.parse_args()
num_of_iterations = 500
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

dst = datasets.CIFAR100("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = args.index


def test_image(img_index,learning_iterations = 0,epsilon = 0):
    gt_data = tp(dst[img_index][0]).to(device)
    if len(args.image) > 1:
        gt_data = Image.open(args.image)
        gt_data = tp(gt_data).to(device)
    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)

    # plt.imshow(tt(gt_data[0].cpu()))

    from models.vision import LeNet, weights_init
    model = LeNet().to(device)


    torch.manual_seed(1234)

    model.apply(weights_init)
    criterion = cross_entropy_for_onehot
    #################### Train ####################
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    iters, losses = [], []
    n = 0

    while n<learning_iterations:
        y = model(gt_data)
        loss = criterion(y, gt_onehot_label)
        # dy_dx = torch.autograd.grad(y, net.parameters())
        optimizer.zero_grad()
        loss.backward()  # Backward pass to compute the gradient of loss w.r.t our learnable params.
        optimizer.step()  # Update params
        # save the current training information
        # iters.append(n)
        # losses.append(float(loss))  # compute average loss
        n += 1
        if n % 100 == 0:
            print("Iteration: {0} Loss: {1} ".format(n, loss))

    ########################################
    # compute original gradient
    pred = model(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, model.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    #### adding noise!! ####
    #original_dy_dx = [w_layer + torch.normal(mean = 0, std= 0.01,size = w_layer.shape) for w_layer in original_dy_dx]
    #original_dy_dx = [w_layer+np.random.laplace(0,epsilon,w_layer.shape) for w_layer in original_dy_dx]
    laplace_obj = Laplace(loc=0, scale=epsilon)
    original_dy_dx = [w_layer+laplace_obj.sample(w_layer.shape) for w_layer in original_dy_dx]

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    # plt.imshow(tt(dummy_data[0].cpu()))

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


    history = []
    current_loss = torch.Tensor([1])
    iters = 0
    #for iters in range(num_of_iterations):
    while (current_loss.item()>0.003 and iters < 200):

        def closure():
            optimizer.zero_grad()

            dummy_pred = model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
        #     print(iters, "%.4f" % current_loss.item())
        #     history.append(tt(dummy_data[0].cpu()))
        iters = iters + 1
    # plt.figure(figsize=(12, 8))
    # for i in range(round(iters / 10)):
    #     plt.subplot(int(np.ceil(iters / 100)), 10, i + 1)
    #     plt.imshow(history[i])
    #     plt.title("iter=%d" % (i * 10))
    #     plt.axis('off')
    return current_loss.item()



# l = []
# for i in range(10):
#     l.append(test_image(img_index,learning_iterations=500+50*i))
# print(l)
#plt.hist([7 if (x>5) else x for x in l])
# plt.plot(l)
image_number_list = [random.randrange(1, 1000, 1) for i in range(5)]
epsilon_list = [0.1,0.08,0.06,0.03,0.01,0.003,0.001,0.0003,0.0001]
print("chosen images: {0}".format(image_number_list))

def run_dlg_tests(image_number_list,epsilon_list):
    plt.xscale("log")
    loss_per_epsilon_matrix = np.zeros([len(epsilon_list),len(image_number_list)])
    for i,epsilon in enumerate(epsilon_list):
        for j,n in enumerate(image_number_list):
            loss_per_epsilon_matrix[i, j] = test_image(n, learning_iterations=0, epsilon=epsilon)
            #loss_per_epsilon_matrix[i, j] = i+j
        print("epsilon:{0} loss values:{1}".format(epsilon,loss_per_epsilon_matrix[i]))
    with open('../output/epsilon_mat.npy', 'wb') as f:
        np.save(f, loss_per_epsilon_matrix)
    np.savetxt('../output/epsilon_mat.txt', loss_per_epsilon_matrix, fmt='%1.4e')
    plt.plot(epsilon_list,np.mean(loss_per_epsilon_matrix,axis=1))
    plt.title("dlg loss for different levels of laplace noise")
    plt.grid(visible=True,axis="y")
    plt.grid(visible=True,which='minor')
    plt.xlabel("epsilon")
    plt.ylabel("loss")

#print(test_image(30, learning_iterations=0, epsilon=0.01))

run_dlg_tests(image_number_list,epsilon_list)
plt.show()


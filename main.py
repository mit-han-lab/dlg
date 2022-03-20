# -*- coding: utf-8 -*-
import argparse
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot
import random
from torch.distributions.laplace import Laplace
from vision import LeNet, CNN, weights_init
import copy




parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str,default="",
                    help='the path to customized image.')
parser.add_argument('--dataset', type=str, default="CIFAR10",
                    help='pick between - CIFAR100, CIFAR10.')

args = parser.parse_args()
num_of_iterations = 200
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

dst = getattr(datasets, args.dataset)("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = args.index

def noise_function(original_dy_dx,epsilon):
    if (epsilon >0):
        laplace_obj = Laplace(loc=0, scale=epsilon)
        return [w_layer+laplace_obj.sample(w_layer.shape) for w_layer in original_dy_dx]
    return original_dy_dx

def run_dlg(img_index, model=None, train_loader=None, test_loader=None, noise_func = lambda x, y: x, learning_epoches = 0, epsilon=0.1,read_grads=-1,model_number=0):


    gt_data = tp(dst[img_index][0]).to(device)
    if len(args.image) > 1:
        gt_data = Image.open(args.image)
        gt_data = tp(gt_data).to(device)

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)

    #################### Model Configuration ####################

    model = LeNet().to(device)

    torch.manual_seed(1234)
    model.apply(weights_init)
    criterion = cross_entropy_for_onehot
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if (read_grads == -1):# run the original images
        #################### Train & Test ####################
        if (learning_epoches >0):
            model.train_nn(train_loader=train_loader, optimizer=optimizer, criterion=criterion,  epoch_num=learning_epoches,test_loader=test_loader)
            model.test_nn(test_loader,criterion)
        ######################################################
        # compute original gradient
        pred = model(gt_data)
        y = criterion(pred, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, model.parameters())
    else: # get the images from the fed-learn
        grad_checkpoint_address = "./fed-ler_checkpoints/grad/checkpoint{0}_{1}.pk".format(model_number,read_grads)
        global_checkpoint_address = "./fed-ler_checkpoints/global/checkpoint{0}_{1}.pk".format(model_number,read_grads)
        import sys
        sys.path.append(r"C:\Users\tomer\Documents\Final_project_git\federated_learning_uveqfed_dlg\Federated-Learning-Natalie")
        from models import LENETLayer
        fed_ler_grad_state_dict = torch.load(grad_checkpoint_address)


        global_model = torch.load(global_checkpoint_address)
        model =global_model
        # luckily the state dict is saved in exactly the same order as the gradients are so we can easily transfer them
        dy_dx = tuple([fed_ler_grad_state_dict[key] for key in fed_ler_grad_state_dict.keys()])
    if (epsilon > 0):
        original_dy_dx = noise_func(list((_.detach().clone() for _ in dy_dx)), epsilon)
    else:
        original_dy_dx = dy_dx
    #### adding noise!! ####
    #original_dy_dx = [w_layer + torch.normal(mean = 0, std= 0.01,size = w_layer.shape) for w_layer in original_dy_dx]
    #original_dy_dx = [w_layer+np.random.laplace(0,epsilon,w_layer.shape) for w_layer in original_dy_dx]


    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    # plt.imshow(tt(dummy_data[0].cpu()))

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


    history = []
    current_loss = torch.Tensor([1])
    iters = 0
    #for iters in range(num_of_iterations):
    while (iters < num_of_iterations):
    # while (current_loss.item()>0.001 and iters < num_of_iterations):

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
            print(iters, "%.4f" % current_loss.item())
            history.append(tt(dummy_data[0].cpu()))
        iters = iters + 1
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(tt(dummy_data[0].cpu()))

    plt.subplot(1, 2, 2)
    plt.imshow(dst[img_index][0])
    plt.axis('off')

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


import iDLG

def run_epsilon_dlg_idlg_tests(image_number_list,epsilon_list, algo='DLG'):
    plt.xscale("log")
    loss_per_epsilon_matrix = np.zeros([len(epsilon_list),len(image_number_list)])
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True,transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    # run all the tests:
    for i,epsilon in enumerate(epsilon_list):
        for j,n in enumerate(image_number_list):
            extract_img = run_dlg if algo == 'DLG' else iDLG.run_idlg
            loss_per_epsilon_matrix[i, j] = extract_img(n,
                                                        train_loader=train_loader,
                                                        test_loader=test_loader,
                                                        learning_epoches=0,
                                                        epsilon=epsilon,
                                                        noise_func=noise_function,
                                                        read_grads=-1,
                                                        model_number=0)
        #loss_per_epsilon_matrix[i, j] = i+j
        print("epsilon:{0} average loss: {1} loss values:{2}".format(epsilon,np.mean(loss_per_epsilon_matrix[i]),loss_per_epsilon_matrix[i]))

    # save the loss into a matrix
    with open('../output/epsilon_mat'+algo+'.npy', 'wb') as f:
        np.save(f, loss_per_epsilon_matrix)
    np.savetxt('../output/epsilon_mat'+algo+'.txt', loss_per_epsilon_matrix, fmt='%1.4e')

    # plot the accuracy
    plt.figure()
    plt.plot(epsilon_list,np.mean(loss_per_epsilon_matrix,axis=1))
    plt.title("dlg loss for different levels of laplace noise")
    plt.grid(visible=True,axis="y")
    plt.grid(visible=True,which='minor')
    plt.xlabel("2/epsilon")
    plt.ylabel("loss")

def run_dlg_idlg_tests(image_number_list,check_point_list,model_number, algo='DLG'):
    plt.xscale("log")
    loss_per_iter_matrix = np.zeros([len(check_point_list),len(image_number_list)])
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True,transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    # run all the tests:
    for i,iter in enumerate(check_point_list):
        for j,n in enumerate(image_number_list):
            extract_img = run_dlg if algo == 'DLG' else iDLG.run_idlg
            loss_per_iter_matrix[i, j] = extract_img(n,
                                                        train_loader=train_loader,
                                                        test_loader=test_loader,
                                                        learning_epoches=0,
                                                        epsilon=0,
                                                        noise_func=noise_function,
                                                        read_grads=iter,
                                                        model_number=model_number)
        #loss_per_epsilon_matrix[i, j] = i+j
        print("iter:{0} average loss: {1} loss values:{2}".format(iter,np.mean(loss_per_iter_matrix[i]),loss_per_iter_matrix[i]))

    # save the loss into a matrix
    with open('../output/loss_mat'+algo+'.npy', 'wb') as f:
        np.save(f, loss_per_iter_matrix)
    np.savetxt('../output/loss_mat'+algo+'.txt', loss_per_iter_matrix, fmt='%1.4e')

    # plot the accuracy
    plt.figure()
    plt.plot(check_point_list,np.mean(loss_per_iter_matrix,axis=1))
    plt.title("dlg loss for different number of iterations")
    plt.grid(visible=True,axis="y")
    plt.grid(visible=True,which='minor')
    plt.xlabel("iter")
    plt.ylabel("loss")


if __name__ == "__main__":
    number_of_images = 1
    # image_number_list = [random.randrange(1, 1000, 1) for i in range(number_of_images)]
    image_number_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,508]
    #image_number_list = [3767]
    # epsilon_list = [0.1,0.08,0.06,0.03,0.01,0.003,0.001,0.0003,0.0001]
    epsilon_list = [0]
    print("chosen images: {0}".format(image_number_list))
    check_point_list = [i for i in range(0,400,100)]
    model_number = 813665
    # run_dlg_idlg_tests(image_number_list,check_point_list,model_number,algo='DLG')
    # run_epsilon_dlg_idlg_tests(image_number_list,epsilon_list,algo='DLG')

    #run_dlg(30, learning_epoches=50, epsilon=0)
    K = 25
    print("image= {0}".format(K))
    # run_epsilon_dlg_idlg_tests([9],[0],'DLG')
    run_dlg(K)
    plt.show()

# -*- coding: utf-8 -*-
import argparse
import numpy as np

import iDLG
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
import sys

tomer_path = r"C:\Users\tomer\Documents\Final_project_git\federated_learning_uveqfed_dlg\Federated-Learning-Natalie"
elad_path = r"/Users/elad.sofer/src/Engineering Project/federated_learning_uveqfed_dlg/Federated-Learning-Natalie"
sys.path.append(elad_path)
sys.path.append(tomer_path)

from models import LENETLayer
from federated_utils import PQclass

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
parser.add_argument('--dataset', type=str, default="CIFAR10",
                    help='pick between - CIFAR100, CIFAR10.')

# Federated learning arguments
parser.add_argument('--R', type=int, default=16,
                    choices=[1, 2, 4],
                    help="compression rate (number of bits)")
parser.add_argument('--epsilon', type=float, default=500,
                    choices=[1, 5, 10],
                    help="privacy budget (epsilon)")
parser.add_argument('--dyn_range', type=float, default=1,
                    help="quantizer dynamic range")
parser.add_argument('--quantization_type', type=str, default='SDQ',
                    choices=[None, 'Q', 'DQ', 'SDQ'],
                    help="whether to perform (Subtractive) (Dithered) Quantization")
parser.add_argument('--quantizer_type', type=str, default='mid-tread',
                    choices=['mid-riser', 'mid-tread'],
                    help="whether to choose mid-riser or mid-tread quantizer")

parser.add_argument('--privacy_noise', type=str, default='laplace',
                    choices=[None, 'laplace', 'PPN'],
                    help="add the signal privacy preserving noise of type laplace or PPN")

parser.add_argument('--device', type=str, default='cpu',
                    choices=['cuda:0', 'cuda:1', 'cpu'],
                    help="device to use (gpu or cpu)")

parser.add_argument('--attack', type=str, default='JOPEQ',
                    choices=['JOPEQ', 'noise_only', 'quantization'],
                    help="DLG/iDLG attack type ")
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


def add_uveqFed(original_dy_dx, epsilon, bit_rate):
    noised_dy_dx = []
    args.epsilon = epsilon
    args.R = bit_rate
    noiser = PQclass(args)
    for g in original_dy_dx:
        if args.attack=='JOPEQ':
            output, dither = noiser(g)
            noised_dy_dx.append(output - dither)
            # output = noiser.apply_quantization(g)
            # noised_dy_dx.append(output)
        elif args.attack=="quantization":
            # quantization only
            output = noiser.apply_quantization(g)
            noised_dy_dx.append(output)
        else: # ppn only
            output = noiser.apply_privacy_noise(g)
            noised_dy_dx.append(output)

    return noised_dy_dx


def run_dlg(img_index, model=None, train_loader=None, test_loader=None, noise_func = lambda x, y, z: x, learning_epoches = 0, epsilon=0.1, bit_rate=1,read_grads=-1,model_number=0):

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
        fed_ler_grad_state_dict = torch.load(grad_checkpoint_address)


        global_model = torch.load(global_checkpoint_address)
        model =global_model
        # luckily the state dict is saved in exactly the same order as the gradients are so we can easily transfer them
        dy_dx = tuple([fed_ler_grad_state_dict[key] for key in fed_ler_grad_state_dict.keys()])
    if (epsilon > 0):
        original_dy_dx = noise_func(list((_.detach().clone() for _ in dy_dx)), epsilon, bit_rate)
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
    # while (iters < num_of_iterations):
    while (current_loss.item()>0.00001 and iters < num_of_iterations):

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
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(dst[img_index][0])
    plt.axis('off')

    plt.figure(figsize=(12, 8))
    for i in range(round(iters / 10)):
        plt.subplot(int(np.ceil(iters / 100)), 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')
    return current_loss.item()



# l = []
# for i in range(10):
#     l.append(test_image(img_index,learning_iterations=500+50*i))
# print(l)
#plt.hist([7 if (x>5) else x for x in l])
# plt.plot(l)




def run_epsilon_dlg_idlg_tests(image_number_list,epsilon_list,bit_rate_lst, algo='DLG'):
    """

    Args:
        image_number_list:
        epsilon_list:
        algo:

    Returns:

    """
    plt.xscale("log")
    loss_per_epsilon_matrix = np.zeros([len(bit_rate_lst), len(epsilon_list),len(image_number_list)])
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True,transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    # run all the tests:
    for k, bit_rate in enumerate(bit_rate_lst):
        for i, epsilon in enumerate(epsilon_list):
            print("#### epsilon {0}".format(epsilon))
            for j,n in enumerate(image_number_list):
                extract_img = run_dlg if algo == 'DLG' else iDLG.run_idlg

                loss_per_epsilon_matrix[k, i, j] = extract_img(n,
                                                            train_loader=train_loader,
                                                            test_loader=test_loader,
                                                            learning_epoches=0,
                                                            epsilon=epsilon,
                                                            bit_rate=bit_rate,
                                                            noise_func=add_uveqFed,
                                                            read_grads=-1,
                                                            model_number=0)
                # loss_per_epsilon_matrix[k,i, j] = k+i+j
            print("bit_rate: {0} epsilon:{1} average loss: {2} loss values:{3}".format(bit_rate, epsilon,np.mean(loss_per_epsilon_matrix[k][i]),loss_per_epsilon_matrix[k][i]))

    # # save the loss into a matrix

    #     np.save(f, loss_per_epsilon_matrix[0,:,:])
    # np.savetxt('output/epsilon_mat'+algo+'.txt', loss_per_epsilon_matrix[0,:,:], fmt='%1.4e')

    with open('output/TOTAL_MAT'+algo+'.npy', 'wb') as f:
        pickle.dump(loss_per_epsilon_matrix, f)

    # # plot the accuracy
    # plt.figure()
    # font = {'weight': 'bold','size': 16}
    #
    # plt.rc('font', **font)
    # plt.plot(epsilon_list,np.mean(loss_per_epsilon_matrix,axis=1),linewidth=3)
    # plt.title("{0} loss attack type {1} for various levels of noise levels".format(algo, args.attack))
    # plt.grid(visible=True,axis="y")
    # plt.grid(visible=True,which='minor')
    # plt.xlabel("2/epsilon")
    # plt.ylabel("loss")
import pickle
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
                                                        noise_func=add_uveqFed,
                                                        read_grads=iter,
                                                        model_number=model_number)
        #loss_per_epsilon_matrix[i, j] = i+j
        print("iter:{0} average loss: {1} loss values:{2}".format(iter,np.mean(loss_per_iter_matrix[i]),loss_per_iter_matrix[i]))

    # # save the loss into a matrix
    # with open('../output/loss_mat'+algo+'.npy', 'wb') as f:
    #     np.save(f, loss_per_iter_matrix)
    # np.savetxt('../output/loss_mat'+algo+'.txt', loss_per_iter_matrix, fmt='%1.4e')

    # plot the accuracy
    plt.figure()
    font = {
        'weight': 'bold',
        'size': 16}

    plt.rc('font', **font)
    plt.plot(check_point_list,np.mean(loss_per_iter_matrix,axis=1),linewidth=3)
    plt.title("{0} loss attack type {1}".format(algo, args.attack))
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
    # [0.1, 0.08, 0.06, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    # imagen ids, epsilon list,
    epsilon_lst = [10,33,100,333,1000,3333,10000,100000]
    bit_rate_lst = [4,8,16,32]

    img_lst = list(range(30,45))
    # run_epsilon_dlg_idlg_tests(,[0.1,0.08,0.06,0.03,0.01,0.003,0.001,0.0003,0.0001],'DLG')
    run_epsilon_dlg_idlg_tests(img_lst, epsilon_lst, bit_rate_lst=bit_rate_lst, algo=  'DLG')
    # run_epsilon_dlg_idlg_tests([9],[0.0003,0.0001],'DLG')

    # run_dlg(K)
    plt.show()
    pass




# plt.figure()
# font = {'weight': 'bold','size': 16}
# plt.rc('font', **font)
# bit_rate_lst = [2,4,8,16,32,64,128]
# with open("/Users/elad.sofer/src/Engineering Project/dlg/output/epsilon_mat_quant_onlyDLG.npy", "rb") as fd:
#     mat = np.load(fd)
#
#     plt.plot(bit_rate_lst[:6], np.mean(mat[:6],axis=1), 'r-*')
#     # plt.xscale("log")
#
#     plt.title("Quantization only DLG attack vs. compression rate")
#     plt.grid(visible=True,axis="y")
#     plt.grid(visible=True,which='minor')
#     plt.xlabel("compression rate (#bit number per level)")
#     plt.ylabel("loss")
#
#     bit_rate_lst = [2, 4, 8, 16, 32, 64, 128
#                     ]
# with open(
#         "/Users/elad.sofer/src/Engineering Project/dlg/output/epsilon_mat_DITH_QUANTDLG.npy",
#         "rb") as fd:
#     mat = np.load(fd)
#
#     plt.plot(epsilon_lst, np.mean(mat[:5], axis=1), '-*', linewidth=0.5)
#     plt.xscale("log")
#
#     plt.title("JoPEQ DLG attack vs. noise levels")
#     plt.grid(visible=True, axis="y")
#     plt.grid(visible=True, which='minor')
#     plt.xlabel("2/epsilon")
#     plt.ylabel("loss")

# plt.figure()
# font = {'weight': 'bold','size': 16}
# plt.rc('font', **font)
# epsilont_lst = [10,100,1000,10000,100000]
# bit_rate_lst = [2,4,8,16,32]
# with open("/Users/elad.sofer/src/Engineering Project/dlg/output/TOTAL_MATDLG.npy", "rb") as fd:
#     mat = pickle.load(fd)
#
# for k in range(0,mat.shape[0]):
#     plt.plot([2/e for e in epsilont_lst], np.mean(mat[k,:,:],axis=1), '-*')
#     plt.xscale("log")
#     plt.yscale("log")
#
#     plt.title("JoPEQ DLG attack vs. noise levels")
#     plt.grid(visible=True,axis="y")
#     plt.grid(visible=True,which='minor')
#     plt.xlabel("2/epsilon")
#     plt.ylabel("loss")
#
# plt.legend(["4compressionRate", "8compressionRate", "16compressionRate", "32compressionRate"])
pass
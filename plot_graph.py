from matplotlib import pyplot as plt
import numpy as np
import pickle
plt.figure()
font = {'weight': 'bold','size': 16}
plt.rc('font', **font)
epsilon_lst = [10, 33, 100, 333, 1000, 3333, 10000, 100000]
bit_rate_lst = [4, 8, 16, 32]
with open("output/TOTAL_MATDLG.npy", "rb") as fd:
    mat = pickle.load(fd)

for k in range(0,mat.shape[0]):
    plt.plot([2/e for e in epsilon_lst], np.mean(mat[k,:,:],axis=1), '-*')
    plt.xscale("log")
    plt.yscale("log")

    plt.title("JoPEQ DLG attack vs. noise levels")
    plt.grid(visible=True,axis="y")
    plt.grid(visible=True,which='minor')
    plt.xlabel("2/epsilon")
    plt.ylabel("loss")

plt.legend(["4compressionRate", "8compressionRate", "16compressionRate", "32compressionRate"])
plt.show()
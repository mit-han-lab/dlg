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
# mat = np.delete(mat,11,2)
# mat = np.delete(mat,10,2)
# mat = np.delete(mat,9,2)
# mat = np.delete(mat,7,2)
# mat = np.delete(mat,1,2)
for k in range(0,mat.shape[0]):
    plt.plot([2/e for e in epsilon_lst], np.mean(np.log(mat[k,:,:]),axis=1), '-*')
    plt.xscale("log")
    # plt.yscale("log")

    plt.title("JoPEQ DLG attack vs. noise levels")
    plt.grid(visible=True,axis="y")
    plt.grid(visible=True,which='minor')
    plt.xlabel("2/epsilon")
    plt.ylabel("loss")

plt.legend(["4compressionRate", "8compressionRate", "16compressionRate", "32compressionRate"])


plt.figure()
font = {'weight': 'bold','size': 16}
plt.rc('font', **font)

average_im_loss = np.zeros(mat.shape[2])
for j in range(0,mat.shape[2]):
    average_im_loss[j] = np.mean(mat[:,:,j])
print(average_im_loss)
# plt.plot([2/e for e in epsilon_lst], np.mean(mat[:,:,:],axis=1), '-*')
# plt.xscale("log")
# plt.yscale("log")
# for k in range(0,mat.shape[0]):
#     plt.plot([2/e for e in epsilon_lst], np.mean(mat[k,:,:],axis=1), '-*')
#     plt.xscale("log")
    # plt.yscale("log")

#     plt.title("JoPEQ DLG attack vs. noise levels")
#     plt.grid(visible=True,axis="y")
#     plt.grid(visible=True,which='minor')
#     plt.xlabel("2/epsilon")
#     plt.ylabel("loss")
#
# plt.legend(["4compressionRate", "8compressionRate", "16compressionRate", "32compressionRate"])
plt.show()
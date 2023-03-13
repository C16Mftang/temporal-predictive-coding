import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('ggplot')
from src.models import NeuralKalmanFilter, KalmanFilter
from src.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# hyper parameters
seq_len = 2000
iter_reso = 2
lr_reso = 0.02
inf_iterss = np.arange(2, 22, iter_reso)
inf_lrs = np.arange(0.12, 0.32+lr_reso, lr_reso).round(2)
seeds = range(10)

# to store the MSEs
MSE_NKF = np.zeros((len(seeds), len(inf_lrs), len(inf_iterss)))
MSE_KF = np.zeros((len(seeds), 1))

for i_seed, seed in enumerate(seeds):
    print(f'seed: {seed}')
    # fixed values
    # transition matrix A
    dt = 0.001
    A = torch.tensor([[1., dt, 0.5 * dt**2],
                        [0., 1., dt],
                        [0., 0., 1.]]).to(device)

    # random emissin matrix C
    # use the same seed for all Cs to reduce variation
    g_C = torch.Generator()
    g_C.manual_seed(10)
    C = torch.randn((3, 3), generator=g_C).to(device)

    # control input matrix B
    B = torch.tensor([0., 0., 1.]).to(device).reshape((3, 1))

    # initial true dynamics
    z = torch.tensor([0., 0., 0.]).to(device).reshape((3, 1))

    # control input
    def u_fun(t):
        return torch.tensor(np.exp(-0.01 * t)).reshape((1, 1)).to(device, torch.float)

    # noise covariances in KF
    Q = torch.eye(3).to(device)
    R = torch.eye(3).to(device)
    # generating dataset with varying seed for noises
    g_noise = torch.Generator()
    g_noise.manual_seed(seed)
    us = []
    zs = []
    xs = []
    for i in range(seq_len):
        u = u_fun(i)
        z = torch.matmul(A, z) + torch.matmul(B, u) + torch.randn((3, 1), generator=g_noise).to(device)
        x = torch.matmul(C, z) + torch.randn((3, 1), generator=g_noise).to(device)
        us.append(u)
        zs.append(z)
        xs.append(x)

    us = torch.cat(us, dim=1)
    zs = torch.cat(zs, dim=1)
    xs = torch.cat(xs, dim=1)

    zs = to_np(zs)

    # KF
    kf = KalmanFilter(A, B, C, Q, R, latent_size=3)
    zs_kf, _ = kf.inference(xs, us)
    # compute error
    zs_kf = to_np(zs_kf)
    mse = np.mean((zs - zs_kf)**2)
    MSE_KF[i_seed] = mse

    # NKF with various lrs
    for i_lr, inf_lr in enumerate(inf_lrs):
        for i_iter, inf_iters in enumerate(inf_iterss):
            print(f'\t NKF inf lr: {inf_lr}; inf iter: {inf_iters}')
            
            # estimating using NKF, 5 gradient steps
            nkf = NeuralKalmanFilter(A, B, C, latent_size=3, dynamic_inf=False)
            zs_nkf, _ = nkf.predict(xs, us, inf_iters=inf_iters, inf_lr=inf_lr)
            # compute error
            zs_nkf = to_np(zs_nkf)
            mse = np.mean((zs - zs_nkf)**2)
            MSE_NKF[i_seed, i_lr, i_iter] = mse

# visualization

mean_kf = np.mean(MSE_KF)
mean_nkf = np.mean(MSE_NKF, axis=0) # this is a matrix, size=[len(lrs), len(iters)]
diff = mean_nkf - mean_kf
print(diff)

# print(mean_nkf)
# print(mean_kf)

result_path = os.path.join('./results/', 'inference_comparisons')
if not os.path.exists(result_path):
    os.makedirs(result_path)

# an alternative way of visualization
yticks = np.arange(0, 0.8, 0.2)
cmap = mpl.cm.get_cmap('Blues')(np.linspace(0.2, 1, len(inf_iterss)))
fig, ax = plt.subplots(1, 1, figsize=(4,3))
for i in range(len(inf_iterss)):
    ax.plot(diff[:, i], label=inf_iterss[i], c=cmap[i])
ax.set_xticks(np.arange(0, len(inf_lrs), 2), inf_lrs[::2])
ax.set_yscale('log')
ax.set_yticks([1, 2, 3, 4, 5, 6], ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0'])
ax.set_xlabel('Step size')
ax.set_ylabel('Difference')
ax.legend(title='# inference steps', prop={'size': 7}, ncol=2)
ax.set_title('MSE difference between PC and KF', fontsize=12)
plt.tight_layout()
plt.savefig(result_path + '/MSE_curve.pdf')


# need to transpose the matrix to fit the order of lrs and iters!
# diff = np.flip(diff, axis=0)
# plt.figure(figsize=(10, 10))
# plt.imshow(diff)
# for (x, y), value in np.ndenumerate(diff):
#     plt.text(y, x, f"{value:.4f}", va="center", ha="center", c='white')
# # plt.plot(np.arange(len(inf_iterss))+0.5, min_ind, c='red')
# plt.xticks(np.arange(len(inf_iterss)), inf_iterss)
# plt.yticks(np.arange(len(inf_lrs)), np.flip(inf_lrs))
# plt.xlabel('Inference steps')
# plt.ylabel('Step size')
# plt.title('Difference between PC and Kalman Filter MSEs')
# plt.grid(None)
# plt.colorbar()
# plt.savefig(result_path + '/MSE_spectrum', dpi=200)

    
        
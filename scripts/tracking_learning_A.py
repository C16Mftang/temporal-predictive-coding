import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from src.models import NeuralKalmanFilter, KalmanFilter
from src.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

result_path = os.path.join('./results/', 'learning_comparisons')
if not os.path.exists(result_path):
    os.makedirs(result_path)

# hyper parameters
seq_len = 1000
inf_iters = 20
inf_lr = 0.05
learn_iters = 1
learn_lr = 5e-5
seeds = range(10)

latent_mses = np.zeros((3, len(seeds)))
obs_mses = np.zeros((3, len(seeds)))
for ind, seed in enumerate(seeds):
    print(f'seed: {seed} for noise')
    # create the dataset of the tracking problem
    # transition matrix A
    dt = 1e-3
    A = torch.tensor([[1., dt, 0.5 * dt**2],
                    [0., 1., dt],
                    [0., 0., 1.]]).to(device)

    # random emissin matrix C
    g_C = torch.Generator()
    g_C.manual_seed(10)
    C = torch.randn((3, 3), generator=g_C).to(device)
    print(C)

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

    # generating dataset
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

    # estimating using NKF
    nkf = NeuralKalmanFilter(A, B, C, latent_size=3, dynamic_inf=False, learn_transition=False)
    zs_nkf = nkf.train(xs, us, inf_iters, inf_lr)
    print(zs_nkf.shape)

    # learning A, with NKF
    g_A = torch.Generator()
    g_A.manual_seed(1) 
    init_A = torch.randn((3, 3), generator=g_A)
    A_nkf = NeuralKalmanFilter(init_A, B, C, latent_size=3, dynamic_inf=False, learn_transition=True)
    zs_A_nkf = A_nkf.train(xs, us, inf_iters, inf_lr, learn_iters, learn_lr)
    print(zs_A_nkf.shape)

    # random A, with NKF
    rA_nkf = NeuralKalmanFilter(init_A, B, C, latent_size=3, dynamic_inf=False, learn_transition=False)
    zs_rA_nkf = rA_nkf.train(xs, us, inf_iters, inf_lr, learn_iters, learn_lr)
    print(zs_rA_nkf.shape)

    # estimating using KF
    kf = KalmanFilter(A, B, C, Q, R, latent_size=3)
    zs_kf = kf.inference(xs, us)
    print(zs_kf.shape)

    # error on the observation level, emitted by C
    obs_mses[0, ind] = np.mean(to_np(nkf.exs)**2)
    obs_mses[1, ind] = np.mean(to_np(A_nkf.exs)**2)
    obs_mses[2, ind] = np.mean(to_np(rA_nkf.exs)**2)

    zs = to_np(zs)
    zs_nkf = to_np(zs_nkf)
    zs_kf = to_np(zs_kf)
    zs_A_nkf = to_np(zs_A_nkf)
    zs_rA_nkf = to_np(zs_rA_nkf)
    mse = np.mean((zs - zs_nkf)**2)
    mse_A = np.mean((zs - zs_A_nkf)**2)
    mse_rA = np.mean((zs - zs_rA_nkf)**2)
    latent_mses[0, ind] = mse
    latent_mses[1, ind] = mse_A
    latent_mses[2, ind] = mse_rA

# visualize dynamics
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
lw = 0.8
ax.plot(zs_kf[0], label='Kalman Filter', lw=2.5*lw)
ax.plot(zs_nkf[0], label='True A', lw=1.2*lw)
ax.plot(zs_A_nkf[0], label='Learnt A', lw=lw, c='#BDE038')
ax.plot(zs_rA_nkf[0], label='Random A', lw=lw, c='#708A83')
ax.plot(zs[0], label='True Value', c='k', ls='--', lw=lw)
ax.set_title('Position')
ax.legend(prop={'size': 8})
ax.set_xlabel('Time')
# ax.set_ylabel('Value')
plt.tight_layout()
plt.savefig(result_path + f'/learning_A_{inf_iters}_pos.pdf')

# errors on the latent level
latent_mses = np.log10(latent_mses)
xticks = ['True A', 'Learnt A', 'Random A']
mse_mean = np.mean(latent_mses, axis=1)
mse_std = np.std(latent_mses, axis=1)
bar_ticks = np.arange(3)
plt.figure(figsize=(4, 3))
plt.bar(bar_ticks, mse_mean)
plt.errorbar(bar_ticks, mse_mean, yerr=mse_std, c='k', ls='none')
plt.xticks(bar_ticks, xticks)
# plt.ylabel('State MSE')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
plt.title('State MSE (log-scaled)')
plt.tight_layout()
plt.savefig(result_path + f'/mse_comparison_learningA.pdf')

# errors on the observation level
obs_mses = np.log10(obs_mses)
obs_mse_mean = np.mean(obs_mses, axis=1)
obs_mse_std = np.std(obs_mses, axis=1)
bar_ticks = np.arange(3)
plt.figure(figsize=(4, 3))
plt.bar(bar_ticks, obs_mse_mean)
plt.errorbar(bar_ticks, obs_mse_mean, yerr=obs_mse_std, c='k', ls='none')
plt.xticks(bar_ticks, xticks)
# plt.ylabel('Observation MSE')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
plt.title('Observation MSE (log-scaled)')
plt.tight_layout()
plt.savefig(result_path + f'/obs_mse_comparison_learningA.pdf')

print(obs_mse_mean, mse_mean)



import os
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from src.models import NeuralKalmanFilter, KalmanFilter
from src.utils import *

device = "cpu"
print(device)

result_path = os.path.join('./results/', 'learning_comparisons')
if not os.path.exists(result_path):
    os.makedirs(result_path)

# hyper parameters
seq_len = 1000
inf_iters = 20
inf_lr = 0.05
learn_iters = 80
learn_lr = 2e-5
seeds = range(20)

latent_mses = np.zeros((4, len(seeds)))
obs_mses = np.zeros((4, len(seeds)))
start_time = time.time()
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
    g_C.manual_seed(1)
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

    # generate random A and C for initial weights
    g_A = torch.Generator()
    g_A.manual_seed(1) 
    init_A = torch.randn((3, 3), generator=g_A).to(device)

    g_C = torch.Generator()
    g_C.manual_seed(2) # dont' use seed=10 here! It will generate the real C
    init_C = torch.randn((3, 3), generator=g_C).to(device)

    # true A C
    print('True A C')
    nkf = NeuralKalmanFilter(A, B, C, latent_size=3).to(device)
    zs_nkf, xs_nkf = nkf.predict(xs, us, inf_iters, inf_lr)
    
    # learn A C
    print('Learnt A C')
    AC_nkf = NeuralKalmanFilter(init_A, B, init_C, latent_size=3).to(device)
    AC_nkf.train(xs, us, inf_iters, inf_lr, learn_iters, learn_lr)
    zs_AC_nkf, xs_AC_nkf = AC_nkf.predict(xs, us, inf_iters, inf_lr)

    # random A C
    print('Random A C')
    rAC_nkf = NeuralKalmanFilter(init_A, B, init_C, latent_size=3).to(device)
    zs_rAC_nkf, xs_rAC_nkf = rAC_nkf.predict(xs, us, inf_iters, inf_lr)

    # estimating using KF
    print('Kalman filter')
    kf = KalmanFilter(A, B, C, Q, R, latent_size=3).to(device)
    zs_kf, xs_kf = kf.inference(xs, us)

    # error on the observation level, emitted by C
    obs_mses[0, ind] = np.mean(to_np((xs - xs_kf)**2))
    obs_mses[1, ind] = np.mean(to_np((xs - xs_nkf)**2))
    obs_mses[2, ind] = np.mean(to_np((xs - xs_AC_nkf)**2))
    obs_mses[3, ind] = np.mean(to_np((xs - xs_rAC_nkf)**2))

    # error of the latent infered state
    latent_mses[0, ind] = np.mean(to_np((zs - zs_kf)**2))
    latent_mses[1, ind] = np.mean(to_np((zs - zs_nkf)**2))
    latent_mses[2, ind] = np.mean(to_np((zs - zs_AC_nkf)**2))
    latent_mses[3, ind] = np.mean(to_np((zs - zs_rAC_nkf)**2))

print(f'Code finishes, total time: {time.time() - start_time} seconds')

# visualize latent dynamics
dim = 2
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
lw = 0.8
# latent
ax[0].plot(to_np(zs_kf[dim]), label='Kalman Filter', lw=2.5*lw)
ax[0].plot(to_np(zs_nkf[dim]), label='True', lw=1.2*lw)
ax[0].plot(to_np(zs_AC_nkf[dim]), label='Learnt', lw=lw, c='#BDE038')
ax[0].plot(to_np(zs_rAC_nkf[dim]), label='Random', lw=lw, c='#708A83')
ax[0].plot(to_np(zs[dim]), label='True Value', c='k', ls='--', lw=lw)
ax[0].set_title('State', fontsize=12)
ax[0].legend(prop={'size': 7})
ax[0].set_xlabel('Time')
ax[0].set_ylabel(r'$x_3$')
# observed
ax[1].plot(to_np(xs_kf[dim]), label='Kalman Filter', lw=2.5*lw)
ax[1].plot(to_np(xs_nkf[dim]), label='True', lw=1.2*lw)
ax[1].plot(to_np(xs_AC_nkf[dim]), label='Learnt', lw=lw, c='#BDE038')
ax[1].plot(to_np(xs_rAC_nkf[dim]), label='Random', lw=lw, c='#708A83')
ax[1].plot(to_np(xs[dim]), label='True Value', c='k', ls='--', lw=lw)
ax[1].set_title('Observed', fontsize=12)
ax[1].set_xlabel('Time')
ax[1].set_ylabel(r'$y_3$')
plt.tight_layout()
plt.savefig(result_path + f'/learning_AC_inf{inf_iters}_learn{learn_iters}_{dim}.pdf')

# visualize the errors on the latent level
fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
xticks = ['Kalman Filter', 'True', 'Learnt', 'Random']
print(latent_mses.shape)
mse_mean = np.mean(latent_mses, axis=1)
mse_std = np.std(latent_mses, axis=1)
bar_ticks = np.arange(1, 5)
ax[0].boxplot(latent_mses.T, patch_artist=True, boxprops=dict(facecolor='red'), medianprops=dict(color='k'))
ax[0].set_xticks(bar_ticks, xticks)
ax[0].set_yscale('log')
ax[0].set_ylabel('MSE')
ax[0].set_title('State MSE (log-scaled)', fontsize=12)

# visualize errors on the observation level
obs_mse_mean = np.mean(obs_mses, axis=1)
obs_mse_std = np.std(obs_mses, axis=1)
ax[1].boxplot(obs_mses.T, patch_artist=True, boxprops=dict(facecolor='red'), medianprops=dict(color='k'))
ax[1].set_xticks(bar_ticks, xticks)
ax[1].set_yscale('log')
ax[1].set_title('Observation MSE (log-scaled)', fontsize=12)
plt.tight_layout()
plt.savefig(result_path + f'/mse_comparison_learningAC_inf{inf_iters}_learn{learn_iters}.pdf')

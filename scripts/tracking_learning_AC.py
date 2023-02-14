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
learn_lr = 2e-5
seeds = range(10)

latent_mses = np.zeros((4, len(seeds)))
obs_mses = np.zeros((4, len(seeds)))
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
    init_A = torch.randn((3, 3), generator=g_A)

    g_C = torch.Generator()
    g_C.manual_seed(2) # dont' use seed=10 here! It will generate the real C
    init_C = torch.randn((3, 3), generator=g_C)

    # true A C
    nkf = NeuralKalmanFilter(A, B, C, latent_size=3, learn_transition=False, learn_emission=False)
    zs_nkf = nkf.train(xs, us, inf_iters, inf_lr, learn_iters, learn_lr)
    xs_nkf = to_np(nkf.pred_xs)
    print(xs_nkf.shape)

    # learn A C
    AC_nkf = NeuralKalmanFilter(init_A, B, init_C, latent_size=3, learn_transition=True, learn_emission=True)
    zs_AC_nkf = AC_nkf.train(xs, us, inf_iters, inf_lr, learn_iters, learn_lr)
    xs_AC_nkf = to_np(AC_nkf.pred_xs)
    print(xs_AC_nkf.shape)

    # random A C
    rAC_nkf = NeuralKalmanFilter(init_A, B, init_C, latent_size=3, learn_transition=False, learn_emission=False)
    zs_rAC_nkf = rAC_nkf.train(xs, us, inf_iters, inf_lr, learn_iters, learn_lr)
    xs_rAC_nkf = to_np(rAC_nkf.pred_xs)
    print(xs_rAC_nkf.shape)

    # estimating using KF
    kf = KalmanFilter(A, B, C, Q, R, latent_size=3)
    zs_kf = kf.inference(xs, us)
    xs_kf = to_np(kf.pred_xs)
    print(xs_kf.shape)

    # error on the observation level, emitted by C
    obs_mses[0, ind] = np.mean(to_np(kf.exs)**2)
    obs_mses[1, ind] = np.mean(to_np(nkf.exs)**2)
    obs_mses[2, ind] = np.mean(to_np(AC_nkf.exs)**2)
    obs_mses[3, ind] = np.mean(to_np(rAC_nkf.exs)**2)

    # error of the latent infered state
    zs = to_np(zs)
    zs_kf = to_np(zs_kf)
    zs_nkf = to_np(zs_nkf)
    zs_AC_nkf = to_np(zs_AC_nkf)
    zs_rAC_nkf = to_np(zs_rAC_nkf)

    latent_mses[0, ind] = np.mean((zs - zs_kf)**2)
    latent_mses[1, ind] = np.mean((zs - zs_nkf)**2)
    latent_mses[2, ind] = np.mean((zs - zs_AC_nkf)**2)
    latent_mses[3, ind] = np.mean((zs - zs_rAC_nkf)**2)

# visualize latent dynamics
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
lw = 0.8
# latent
ax[0].plot(zs_kf[0], label='Kalman Filter', lw=2.5*lw)
ax[0].plot(zs_nkf[0], label='True', lw=1.2*lw)
ax[0].plot(zs_AC_nkf[0], label='Learnt', lw=lw, c='#BDE038')
ax[0].plot(zs_rAC_nkf[0], label='Random', lw=lw, c='#708A83')
ax[0].plot(zs[0], label='True Value', c='k', ls='--', lw=lw)
ax[0].set_title('State', fontsize=12)
ax[0].legend(prop={'size': 7})
ax[0].set_xlabel('Time')
ax[0].set_ylabel(r'$x_1$')
# observed
ax[1].plot(xs_kf[0], label='Kalman Filter', lw=2.5*lw)
ax[1].plot(xs_nkf[0], label='True', lw=1.2*lw)
ax[1].plot(xs_AC_nkf[0], label='Learnt', lw=lw, c='#BDE038')
ax[1].plot(xs_rAC_nkf[0], label='Random', lw=lw, c='#708A83')
ax[1].plot(xs[0], label='True Value', c='k', ls='--', lw=lw)
ax[1].set_title('Observed', fontsize=12)
# ax[1].legend(prop={'size': 7})
ax[1].set_xlabel('Time')
ax[1].set_ylabel(r'$y_1$')
plt.tight_layout()
plt.savefig(result_path + f'/learning_AC_{inf_iters}.pdf')

# visualize the errors on the latent level
fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
xticks = ['Kalman Filter', 'True', 'Learnt', 'Random']
latent_mses = np.log10(latent_mses)
mse_mean = np.mean(latent_mses, axis=1)
mse_std = np.std(latent_mses, axis=1)
bar_ticks = np.arange(4)
ax[0].bar(bar_ticks, mse_mean)
ax[0].errorbar(bar_ticks, mse_mean, yerr=mse_std, c='k', ls='none')
ax[0].set_xticks(bar_ticks, xticks)
ax[0].set_ylabel('Log MSE')
ax[0].set_title('State MSE (log-scaled)', fontsize=12)

# visualize errors on the observation level
obs_mses = np.log10(obs_mses)
obs_mse_mean = np.mean(obs_mses, axis=1)
obs_mse_std = np.std(obs_mses, axis=1)
ax[1].bar(bar_ticks, obs_mse_mean)
ax[1].errorbar(bar_ticks, obs_mse_mean, yerr=obs_mse_std, c='k', ls='none')
ax[1].set_xticks(bar_ticks, xticks)
ax[1].set_title('Observation MSE (log-scaled)', fontsize=12)
# plt.ylabel('Observation MSE')
# plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
# plt.yscale('log')
plt.tight_layout()
plt.savefig(result_path + f'/mse_comparison_learningAC.pdf')




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

# hyper parameters
seq_len = 1000
inf_iters = 20
inf_lr = 0.1
learn_lr = 1e-5

# create the dataset of the tracking problem
# transition matrix A
dt = 1e-3
A = torch.tensor([[1., dt, 0.5 * dt**2],
                  [0., 1., dt],
                  [0., 0., 1.]]).to(device)

# random emissin matrix C
g_C = torch.Generator()
g_C.manual_seed(200)
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
g_noise.manual_seed(10)
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

# estimating using NKF, 5 gradient steps
nkf = NeuralKalmanFilter(A, B, C, latent_size=3, dynamic_inf=False)
zs_nkf, _ = nkf.predict(xs, us, inf_iters=10, inf_lr=inf_lr)
print(zs_nkf.shape)

# estimating using NKF, 1 gradient steps
nkf1 = NeuralKalmanFilter(A, B, C, latent_size=3, dynamic_inf=False)
zs_nkf1, _ = nkf1.predict(xs, us, inf_iters=1, inf_lr=inf_lr)
print(zs_nkf1.shape)

# estimating using NKF, using PC equilibrium
nkf0 = NeuralKalmanFilter(A, B, C, latent_size=3, dynamic_inf=False)
zs_nkf0, _ = nkf0.predict(xs, us, inf_iters=0, inf_lr=inf_lr)
print(zs_nkf1.shape)

# estimating using NKF with dynamic inference
# d_nkf = NeuralKalmanFilter(A, B, C, latent_size=3, dynamic_inf=True)
# zs_d_nkf = d_nkf.train(xs, us, inf_iters, inf_lr)
# print(zs_d_nkf.shape)

# estimating using KF
kf = KalmanFilter(A, B, C, Q, R, latent_size=3)
zs_kf, _ = kf.inference(xs, us)
print(zs_kf.shape)

# compare inference iters:
result_path = os.path.join('./results/', 'inference_comparisons')
if not os.path.exists(result_path):
    os.makedirs(result_path)

zs = to_np(zs)
zs_nkf = to_np(zs_nkf)
zs_nkf1 = to_np(zs_nkf1)
zs_nkf0 = to_np(zs_nkf0)
zs_kf = to_np(zs_kf)

# fig, ax = plt.subplots(1, 3, figsize=(10, 3))
lw = 1
# ax[0].plot(zs_kf[0, 500:600], label='Kalman Filter')
# ax[0].plot(zs_nkf[0, 500:600], label='5 Steps')
# ax[0].plot(zs_nkf1[0, 500:600], label='Single Step', c='#BDE038', ls='--')
# ax[0].plot(zs_nkf0[0, 500:600], label='PC Equilibrium', c='#F2BC8D', ls=':')
# ax[0].plot(zs[0, 500:600], label='True Value', c='k', ls='--')
# ax[0].set_title('Position')
# ax[0].set_xticks(np.arange(0, 120, 20), np.arange(500, 620, 20))
# ax[0].legend(prop={'size': 8})

# ax[1].plot(zs_kf[1, 500:600])
# ax[1].plot(zs_nkf[1, 500:600])
# ax[1].plot(zs_nkf1[1, 500:600], c='#BDE038', ls='--')
# ax[1].plot(zs_nkf0[1, 500:600], c='#F2BC8D', ls=':')
# ax[1].plot(zs[1, 500:600], c='k', ls='--')
# ax[1].set_xticks(np.arange(0, 120, 20), np.arange(500, 620, 20))
# ax[1].set_title('Velocity')

plt.figure(figsize=(6, 2))
plt.plot(zs[2, 570:591], c='k', label='True')
plt.plot(zs_kf[2, 570:591], label='Kalman Filter')
plt.plot(zs_nkf1[2, 570:591], c='#13678A', label='tPC (1 step)')
plt.plot(zs_nkf[2, 570:591], c='#45C4B0', label='tPC (10 Steps)')
# plt.plot(zs_nkf0[2, 560:600], c='#9AEBA3', label='PC Equilibrium', ls=':')

plt.xticks(np.arange(0, 30, 10), np.arange(570, 600, 10), color='k')
plt.yticks([98, 100], color='k')
plt.title('Estimated Acceleration')
plt.xlabel('Time', color='k')
plt.ylabel('Value', color='k')
plt.legend(prop={'size': 9}, ncol=2)
# fig.supxlabel('Timestep')
# fig.supylabel('Value')
plt.tight_layout()
plt.savefig(result_path + '/gradient_steps_acc_ccn.pdf')

# compare online inference:
if False:
    zs = to_np(zs)
    zs_nkf = to_np(zs_nkf)
    zs_d_nkf = to_np(zs_d_nkf)
    zs_kf = to_np(zs_kf)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    lw = 1
    # ax[0].plot(zs_kf[0, 500:600], label='KF')
    ax[0].plot(zs_nkf[0], label='normal NKF')
    ax[0].plot(zs_d_nkf[0], label='dynamic NKF', c='#BDE038', ls='--')
    ax[0].plot(zs[0], label='True', c='k', ls='--', lw=0.8)
    ax[0].set_title('Position')
    # ax[0].set_xticks(np.arange(0, 120, 20), np.arange(500, 620, 20))
    ax[0].legend()

    # ax[1].plot(zs_kf[1, 500:600])
    ax[1].plot(zs_nkf[1])
    ax[1].plot(zs_d_nkf[1], c='#BDE038', ls='--')
    ax[1].plot(zs[1], c='k', ls='--', lw=0.8)
    # ax[1].set_xticks(np.arange(0, 120, 20), np.arange(500, 620, 20))
    ax[1].set_title('Velocity')

    # ax[2].plot(zs_kf[2, 500:600])
    ax[2].plot(zs_nkf[2])
    ax[2].plot(zs_d_nkf[2], c='#BDE038', ls='--')
    ax[2].plot(zs[2], c='k', ls='--', lw=0.8)
    # ax[2].set_xticks(np.arange(0, 120, 20), np.arange(500, 620, 20))
    ax[2].set_title('Acceleration')

    fig.supxlabel('Timestep')
    fig.supylabel('Value')
    plt.tight_layout()
    plt.savefig('./results/online_inference', dpi=400)


# visualize data itself
xs = to_np(xs)
us = to_np(us)
# zs = to_np(zs)

plt.figure(figsize=(4,3))
plt.plot(zs[0], label=r'position $(x_1)$')
plt.plot(zs[1], label=r'velocity $(x_2)$')
plt.plot(zs[2], label=r'acceleration $(x_3)$', c='k')
plt.legend(prop={'size': 7})
plt.title('True system state', fontsize=12)
plt.xlabel('Time')
plt.ylabel('Value')
plt.tight_layout()
plt.savefig(result_path + '/true_dynamics.pdf')

plt.figure(figsize=(4,3))
plt.plot(xs[0], label=r'$y_1$')
plt.plot(xs[1], label=r'$y_2$')
plt.plot(xs[2], label=r'$y_3$', c='k')
plt.legend(prop={'size': 7})
plt.title('Projected observations', fontsize=12)
plt.xlabel('Time')
plt.ylabel('Value')
plt.tight_layout()
plt.savefig(result_path + '/observed_dynamics.pdf')




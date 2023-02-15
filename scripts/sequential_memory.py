import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from src.models import TemporalPC
from src.utils import *
from src.get_data import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

result_path = os.path.join('./results/', 'image_sequence')
if not os.path.exists(result_path):
    os.makedirs(result_path)

# hyper parameters
seq_len = 10
inf_iters = 100
inf_lr = 1e-2
learn_iters = 500
learn_lr = 1e-4
latent_size = 256
control_size = 10
sparse_penal = 0
n_cued = 1 # number of cued images
assert(n_cued < seq_len)

# load data
(X, _), (X_test, _) = get_mnist('./data', 
                                sample_size=seq_len, 
                                sample_size_test=seq_len,
                                batch_size=1, 
                                seed=1, 
                                device=device,
                                classes=None)
size = X.shape
flattened_size = size[-1]*size[-2]*size[-3]
xs = X.reshape(-1, flattened_size)
us = torch.zeros((seq_len, control_size)).to(device)
print(xs.shape)

model = TemporalPC(control_size, latent_size, flattened_size, nonlin='tanh').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_lr)

# training
overall_losses, hidden_losses, obs_losses = [], [], []
for learn_iter in range(learn_iters):
    seq_loss, hidden_loss, obs_loss = 0, 0, 0
    prev_z = model.init_hidden().to(device)
    for k in range(seq_len):
        x, u = xs[k:k+1], us[k:k+1]
        optimizer.zero_grad()
        model.inference(inf_iters, inf_lr, x, u, prev_z, sparse_penal)
        model.update_grads(x, u, prev_z)
        optimizer.step()
        prev_z = model.z

        # add up the loss value at each time step
        hidden_loss += model.hidden_loss.item()
        obs_loss += model.obs_loss.item()
        seq_loss += model.hidden_loss.item() + model.obs_loss.item()

    print(f'Iteration {learn_iter+1}, loss {seq_loss}')
    overall_losses.append(seq_loss)
    hidden_losses.append(hidden_loss)
    obs_losses.append(obs_loss)

# cued prediction/inference
print('Cued inference begins')
inf_iters = 5000 # increase inf_iters
test_xs = torch.zeros_like(xs).to(device)
test_xs[:n_cued] = xs[:n_cued]
init_test_xs = test_xs.detach().clone()
prev_z = model.init_hidden().to(device)

hidden_states = []
for k in range(seq_len):
    if k + 1 > n_cued:
        # update the observation layer if we are beyond the point where the true image is given
        model.inference(inf_iters, inf_lr, test_xs[k:k+1], us[k:k+1], prev_z, sparse_penal, update_x=True)
    else:
        # do not update if the true image is given
        model.inference(inf_iters, inf_lr, test_xs[k:k+1], us[k:k+1], prev_z, sparse_penal, update_x=False)
    hidden_states.append(to_np(model.z))
    prev_z = model.z


plt.figure()
plt.plot(overall_losses, label='squared error sum')
plt.plot(hidden_losses, label='squared error hidden')
plt.plot(obs_losses, label='squared error obs')
plt.legend()
plt.savefig(result_path + f'/losses_len{seq_len}_inf{inf_iters}', dpi=150)

fig, ax = plt.subplots(3, seq_len, figsize=(seq_len, 3))
for i in range(seq_len):
    ax[0, i].imshow(to_np(xs[i]).reshape(28, 28))
    ax[0, i].axis('off')
    ax[1, i].imshow(to_np(init_test_xs[i]).reshape(28, 28))
    ax[1, i].axis('off')
    ax[2, i].imshow(to_np(test_xs[i]).reshape(28, 28))
    ax[2, i].axis('off')
plt.savefig(result_path + f'/predicted_samples_len{seq_len}_inf{inf_iters}', dpi=150)

fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
for i in range(seq_len):
    ax[i].imshow(hidden_states[i].reshape(int(np.sqrt(latent_size)), int(np.sqrt(latent_size))))
    ax[i].axis('off')
plt.savefig(result_path + f'/latent_activity_len{seq_len}_inf{inf_iters}', dpi=150)

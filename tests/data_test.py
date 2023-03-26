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

result_path = os.path.join('./results/', 'temporary_results')
if not os.path.exists(result_path):
    os.makedirs(result_path)

# hyper parameters
seq_len = 5
sample_size = 1000
test_size = 1
batch_size = 10
inf_iters = 100
inf_lr = 1e-2
learn_iters = 2
learn_lr = 1e-4
latent_size = 256
flattened_size = 784
control_size = 10
sparse_penal = 0
n_cued = 1 # number of cued images
seed = 2
angle = 30
assert(n_cued < seq_len)

# load data
loader, test_data = get_rotating_mnist('./data', 
                               seq_len, 
                               sample_size,
                               test_size,
                               batch_size,
                               seed, 
                               device, 
                               angle=angle, 
                               test_digit=9)


d = next(iter(loader))
print(d.shape)
d = to_np(d.reshape((batch_size, seq_len, flattened_size)))
fig, ax = plt.subplots(batch_size, seq_len)
for i in range(batch_size):
    for j in range(seq_len):
        ax[i, j].imshow(d[i, j].reshape((28, 28)))
plt.savefig(result_path + f'/examples', dpi=150)
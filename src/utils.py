import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm

def to_np(x):
    return x.cpu().detach().numpy()


def to_torch(x, device):
    return torch.from_numpy(x).to(device, torch.float32)


class Tanh(nn.Module):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0


class Linear(nn.Module):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return torch.ones((1,)).to(inp.device)

class ReLU(nn.Module):
    def forward(self, inp):
        return torch.relu(inp)

    def deriv(self, inp):
        out = self(inp)
        out[out > 0] = 1.0
        return out


def train_batched_input(model, optimizer, scheduler, loader, 
    learn_iters, inf_iters, inf_lr, sparseWout, sparseWr, sparsez, device):
    """Function to train tPC with batched inputs;"""
    train_losses = []
    hidden_losses, obs_losses = [], []
    for learn_iter in range(learn_iters):
        epoch_loss = 0
        hidden_loss, obs_loss = 0, 0
        # train the model
        model.train()
        with tqdm(total=len(loader.dataset)) as pbar:
            for xs in loader:
                # xs = xs[0]
                batch_size, seq_len = xs.shape[:2]

                # reshape image to vector
                xs = xs.reshape((batch_size, seq_len, -1)).to(device)

                # initialize the hidden activities
                prev = model.init_hidden(batch_size).to(device)

                batch_loss = 0
                batch_hidden_loss, batch_obs_loss = 0, 0
                for k in range(seq_len):
                    x = xs[:, k, :].clone().detach()
                    optimizer.zero_grad()
                    model.inference(inf_iters, inf_lr, x, prev, sparsez)
                    energy = model.get_energy(x, prev)

                    # add sparse constraint
                    l1_norm = sparseWout * torch.linalg.norm(model.Wout.weight, 1) + sparseWr * torch.linalg.norm(model.Wr.weight, 1)
                    energy += l1_norm
                        
                    energy.backward()
                    optimizer.step()
                    prev = model.z.clone().detach()

                    # weight normalization - necessary for sparse coding!
                    model.weight_norm()

                    # add up the loss value at each time step then get the average across sequence elements
                    batch_loss += energy.item() / seq_len

                    # notice that they do not add up to the total energy due to the sparse constraint
                    batch_hidden_loss += model.hidden_loss.item() / seq_len
                    batch_obs_loss += model.obs_loss.item() / seq_len

                # add the loss in this batch, and average across batches i.e., this epoch's batch average loss
                epoch_loss += batch_loss / (len(loader.dataset) // batch_size)

                # collect layer-wise losses
                hidden_loss += batch_hidden_loss / (len(loader.dataset) // batch_size)
                obs_loss += batch_obs_loss / (len(loader.dataset) // batch_size)

                # update progress bar
                pbar.set_postfix({'epoch': learn_iter, 'loss': "%.4f" % batch_loss})
                pbar.update(batch_size)

            train_losses.append(epoch_loss)
            hidden_losses.append(hidden_loss)
            obs_losses.append(obs_loss)
            scheduler.step()

    return (train_losses, hidden_losses, obs_losses)

def train_sparse_coding(model, optimizer, scheduler, loader, device, args):
    """Function to train sparse coding with batched frames;"""
    train_losses = []
    for learn_iter in range(args.learn_iters):
        epoch_loss = 0

        # train the model
        model.train()
        with tqdm(total=len(loader.dataset)) as pbar:
            for x in loader:
                # xs: batch_size x input_size
                batch_size = x.shape[0]

                # reshape image to vector
                x = x.reshape((batch_size, -1)).to(device)

                # initialize the hidden activities for each batch
                init_z = model.init_hidden(batch_size).to(device)

                optimizer.zero_grad()
                model.inference(args.inf_iters, args.inf_lr, x, init_z, args.sparsez)
                energy = model.get_energy(x)

                # add sparse constraint
                l1_norm = args.sparseW * torch.linalg.norm(model.Wout.weight, 1)
                # l1_norm /= (x.shape[1] * init_z.shape[1]) 
                energy += l1_norm
                
                # backprop
                energy.backward()
                optimizer.step()

                # weight normalization - necessary for sparse coding!
                model.weight_norm()

                # collect the loss
                batch_loss = energy.item()

                # update progress bar
                pbar.set_postfix({'epoch': learn_iter, 'loss': "%.4f" % batch_loss})
                pbar.update(batch_size)

                # add the loss in this batch, and average across batches i.e., this epoch's batch average loss
                epoch_loss += batch_loss / (len(loader.dataset) // batch_size)

            train_losses.append(epoch_loss)
            scheduler.step()

    return train_losses

def get_strf(hidden, test, tau, device):
    """
    hidden: hidden response of the model; test_size, seq_len, hidden_size
    test: stimuli to input to the model; test_size, seq_len, h*w
    tau: number of preceding frames

    Output:
    strfs: hidden_size, tau, h*w
    """
    hidden_size = hidden.shape[-1]
    seq_len = hidden.shape[1]
    stim_dim = test.shape[-1]

    all_units_strfs = np.zeros((hidden_size, tau, stim_dim))
    for j in range(hidden_size):
        response = hidden[:, :, j].to(device) # test_size, seq_len

        strfs = torch.zeros((tau, stim_dim)).to(device)
        for k in range(tau, seq_len):
            # get the response at the current step
            res = response[:, k].unsqueeze(-1).repeat(1, tau).unsqueeze(-1) # (test_size, tau, 1)

            # weight the preceding stimuli with these response
            preceding_stim = test[:, k-tau+1:k+1] # (test_size, tau, (h*w))
            weighted_preceding_stim = res * preceding_stim # (test_size, tau, (h*w))

            # average the strf along the batch dimension
            strfs += weighted_preceding_stim.mean(dim=0) # (tau, h*w)

        strfs /= (seq_len - tau) # tau, (h*w)

        all_units_strfs[j] = to_np(strfs)
    return all_units_strfs

def spatiotemporal_whitening(data, mode='ZCA'):
    # perform spatio temporal whitening on the data
    # data: test_size, seq_len, h*w
    data = data.reshape(data.shape[0], data.shape[1], -1)
    dim_feature = data.shape[-1]
    # 1. Mean Removal
    mean_feature = np.mean(data, axis=(0, 1)) # shape: h*w
    data_centered = data - mean_feature # shape: test_size, seq_len, h*w

    # 2. Compute the Covariance Matrix
    cov_matrix = np.cov(data_centered.reshape(-1, dim_feature).T) # shape: h*w, h*w

    # 3. Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. Whitening Transformation
    D_inv_sqrt = np.diag(eigenvalues ** (-0.5)) 
    if mode == 'PCA':
        whitening_matrix = np.dot(eigenvectors, D_inv_sqrt)
    elif mode == 'ZCA':
        whitening_matrix = np.dot(np.dot(eigenvectors, D_inv_sqrt), eigenvectors.T)
    elif mode == 'None':
        return data, np.eye(dim_feature)
    else:
        raise ValueError('mode should be either PCA or ZCA or None')

    return np.dot(data_centered, whitening_matrix), whitening_matrix
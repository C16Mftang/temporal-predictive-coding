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

def train_batched_input(model, optimizer, loader, learn_iters, inf_iters, inf_lr, sparseW, sparsez, device):
    """Function to train tPC with batched inputs;"""
    train_losses = []
    start_time = time.time()
    for learn_iter in range(learn_iters):
        epoch_loss = 0

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
                for k in range(seq_len):
                    x = xs[:, k, :].clone().detach()
                    optimizer.zero_grad()
                    model.inference(inf_iters, inf_lr, x, prev, sparsez)
                    energy = model.get_energy(x, prev)

                    # add sparse constraint
                    if sparseW is not None:
                        l1_norm = sum(torch.linalg.norm(p, 1) for p in model.parameters())
                        energy += sparseW * l1_norm
                        
                    energy.backward()
                    optimizer.step()
                    prev = model.z.clone().detach()

                    # weight normalization - necessary for sparse coding!
                    model.weight_norm()

                    # add up the loss value at each time step then get the average across sequence elements
                    batch_loss += energy.item() / seq_len

                # add the loss in this batch, and average across batches i.e., this epoch's batch average loss
                epoch_loss += batch_loss / (len(loader.dataset) // batch_size)

                # update progress bar
                pbar.set_postfix({'epoch': learn_iter, 'loss': "%.4f" % batch_loss})
                pbar.update(batch_size)

            train_losses.append(epoch_loss)
            # val_losses.append(val_loss)
            # if (learn_iter + 1) % 10 == 0:
            #     print(f'Epoch {learn_iter+1}, train loss {epoch_loss}')

    print(f'training PC complete, time: {time.time() - start_time}')
    return train_losses
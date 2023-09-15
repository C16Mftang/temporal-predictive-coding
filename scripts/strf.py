import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import opinionated
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from src.models import TemporalPC, MultilayertPC
from src.utils import *
from src.get_data import get_nat_movie

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# training parameters as command line arguments
parser = argparse.ArgumentParser(description='Spatio-temporal receptive fields')

parser.add_argument('--datapath', type=str,
                    help='path to nat data, must specify')
parser.add_argument('--train-size', type=int, default=100000, 
                    help='training size')
parser.add_argument('--batch-size', type=int, default=10000, 
                    help='training batch size')
parser.add_argument('--hidden-size', type=int, default=1024,
                    help='hidden siz')
parser.add_argument('--learn-lr', type=float, default=2e-4,
                    help='learning rate for PC')
parser.add_argument('--learn-iters', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--inf-lr', type=float, default=1e-2,
                    help='inference step size')
parser.add_argument('--inf-iters', type=int, default=20,
                    help='inference steps in each training epoch')
parser.add_argument('--sparseW', type=float, default=2.0,
                    help='spasity level for model parameters')
parser.add_argument('--sparsez', type=float, default=2.0,
                    help='spasity level for hidden activities')
parser.add_argument('--STA', type=str, default='False', choices=['False', 'True'],
                    help='whether to perform STA')       
parser.add_argument('--std', type=float, default=3.,
                    help='level of standard deviation for white noise')               
parser.add_argument('--tau', type=int, default=6,
                    help='number of preceding frames in STA')      
parser.add_argument('--nonlin', type=str, default='linear', choices=['linear', 'tanh'],
                    help='nonlinearity') 
parser.add_argument('--infer-with', type=str, default='white noise', choices=['white noise', 'test'],
                    help='data on which the inference is performed')

args = parser.parse_args()

def _plot_train_loss(train_losses, result_path):
    # plotting loss for tunning; temporary
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.legend()
    plt.savefig(result_path + f'/train_losses')

def _plot_inf_losses(inf_losses, result_path):
    # plotting loss for tunning; temporary
    plt.figure()
    plt.plot(to_np(inf_losses), label='inf')
    plt.legend()
    plt.xlabel('Inference iters')
    plt.savefig(result_path + f'/inf_losses')

def _plot_strf(all_strfs, tau, result_path, hidden_size, n_files=20):
    n_units_per_file = hidden_size // n_files
    strf_min, strf_max = np.min(all_strfs), np.max(all_strfs)
    for f in range(n_files):
        strfs = all_strfs[f*n_units_per_file:(f+1)*n_units_per_file] # n_units_per_file, tau
        fig, ax = plt.subplots(n_units_per_file, tau, figsize=(tau//2, n_units_per_file//2))
        for i in range(n_units_per_file):
            # normalize the filters
            rf = strfs[i]
            # rf = (rf - np.min(rf)) / (np.max(rf) - np.min(rf))
            # rf = 2 * rf - 1
            strf_min, strf_max = -np.max(np.abs(rf)), np.max(np.abs(rf))
            ax[i, 0].set_ylabel(f'#{(i + 1) + (n_units_per_file * f)}', fontsize=8)
            for j in range(tau):
                ax[i, j].imshow(rf[j], cmap='gray', vmin=strf_min, vmax=strf_max)
                ax[i, j].get_xaxis().set_ticks([])
                ax[i, j].get_yaxis().set_ticks([])
                if i == 0:
                    ax[i, j].set_title(f't - {tau-j}', fontsize=10)
        fig.tight_layout()
        plt.savefig(result_path + f'/strf_group{f+1}')
        plt.close()

def _plot_weights(Wr, Wout, hidden_size, h, w, result_path):
    # plot Wout
    Wout = to_np(Wout)
    Wmin, Wmax = np.min(Wout), np.max(Wout)
    fig, axes = plt.subplots(hidden_size // 32, 32, figsize=(8, (hidden_size // 32) // 4))
    for i, ax in enumerate(axes.flatten()):
        f = Wout[:, i]
        im = ax.imshow(f.reshape((h, w)), cmap='gray', vmin=Wmin, vmax=Wmax)
        ax.axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    # fig.tight_layout()
    plt.savefig(result_path + '/Wout', dpi=200)

    # d = int(np.sqrt(hidden_size))
    # fig, axes = plt.subplots(hidden_size // 32, 32, figsize=(8, 8))
    # for i, ax in enumerate(axes.flatten()):
    #     ax.imshow(to_np(Wr)[:, i].reshape((d, d)), cmap='gray')
    #     ax.axis('off')
    # plt.savefig(result_path + '/Wr')

def main(args):
    h, w = 16, 16
    seq_len = 50

    # hyperparameters
    datapath = args.datapath
    train_size = args.train_size
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    learn_lr = args.learn_lr
    learn_iters = args.learn_iters
    inf_lr = args.inf_lr
    inf_iters = args.inf_iters
    sparseW = args.sparseW
    sparsez = args.sparsez
    nonlin = args.nonlin

    # inference hyperparameters
    STA = args.STA
    std = args.std
    tau = args.tau
    infer_with = args.infer_with

    # initialize model
    tPC = MultilayertPC(hidden_size, h * w, nonlin).to(device)
    # apply lr decay
    optimizer = torch.optim.Adam(tPC.parameters(), lr=learn_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.999)

    # Train model
    if STA == 'False':
        # make directory for saving files
        now = time.strftime('%b-%d-%Y-%H-%M-%S', time.gmtime(time.time()))
        path = f'strf-{now}'
        result_path = os.path.join('./results/', path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # processing data
        train = get_nat_movie(datapath, train_size).reshape((train_size, -1, h, w))
        
        # d_path = "data/nat_data/nat_16x16x50.npy"
        # movie = np.load(d_path, mmap_mode='r+') # mmap to disk?
        # train = movie[:train_size].reshape((train_size, -1, h, w))

        # make training data a dataloader
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        # train model                                
        train_losses = train_batched_input(tPC, optimizer, scheduler, train_loader, learn_iters, inf_iters, inf_lr, sparseW, sparsez, device)
        torch.save(tPC.state_dict(), os.path.join(result_path, f'model.pt'))
        _plot_train_loss(train_losses, result_path)

        # visualize weights learned
        Wout = tPC.Wout.weight # 
        Wr = tPC.Wr.weight
        _plot_weights(Wr, Wout, hidden_size, h, w, result_path)

    # evaluate with white noise
    elif STA == 'True':
        dir = input('Select a model by entering its sub-directory:')
        result_path = os.path.join('./results/', dir)

        if not os.path.exists(result_path):
            raise Exception("Specified model not found!")
        else:
            # load the model
            tPC.load_state_dict(torch.load(os.path.join(result_path, f'model.pt'), 
                                           map_location=torch.device(device)))
            tPC.eval()

            # visualize weights learned
            Wout = tPC.Wout.weight 
            Wr = tPC.Wr.weight
            _plot_weights(Wr, Wout, hidden_size, h, w, result_path)

            test_size = 10000
            test_seq_len = 50
            seq_len = test_seq_len
            # create test data from unseen set
            if infer_with == 'test':
                d_path = "data/nat_data/nat_16x16x50.npy"
                movie = np.load(d_path, mmap_mode='r+') # mmap to disk?
                test = movie[train_size:train_size+test_size]

            # create white noise stimuli
            elif infer_with == 'white noise':
                g = torch.Generator()
                g.manual_seed(1)
                white_noise = (torch.rand((test_size, seq_len, h * w), generator=g) < 0.5).to(device, torch.float32)
                # convert them to -1 and 1s
                white_noise = std * (white_noise * 2 - 1)
                # white_noise = torch.randn((test_size, seq_len, h * w), generator=g).to(device, torch.float32) * std
                test = to_np(white_noise)

            # perform inference on the white noise stimuli
            inf_iters_test = 100
            inf_lr_test = 5e-2

            # initialize the hidden activities; batch size is 1 as we are interested in one sequence only
            prev = tPC.init_hidden(test_size).to(device)

            # avrage hidden activities across test movies
            hidden = torch.zeros((test_size, seq_len, hidden_size)).to(device)

            # run inference on test seqence
            inf_losses = torch.zeros((inf_iters_test, ))
            for k in range(seq_len):
                x = to_torch(test[:, k], device)
                tPC.inference(inf_iters_test, inf_lr_test, x, prev, sparsez)
                prev = tPC.get_hidden()
                hidden[:, k] = tPC.get_hidden()
                inf_losses += tPC.get_inf_losses() / seq_len
            _plot_inf_losses(inf_losses, result_path)

            # iterate through neurons and examine their strfs
            all_units_strfs = np.zeros((hidden_size, tau, h, w))
            test = to_torch(test, device)
            for j in range(hidden_size):
                t1 = time.time()
                response = hidden[:, :, j].to(device) # test_size, seq_len

                strfs = torch.zeros((tau, h * w)).to(device)
                for k in range(tau, seq_len):
                    # get the response at the current step
                    res = response[:, k].unsqueeze(-1).repeat(1, tau).unsqueeze(-1) # (test_size, tau, 1)
    
                    # weight the preceding stimuli with these response
                    preceding_stim = test[:, k-tau:k] # (test_size, tau, (h*w))
                    weighted_preceding_stim = res * preceding_stim # (test_size, tau, (h*w))

                    # average the strf along the batch dimension
                    strfs += weighted_preceding_stim.mean(dim=0) # (tau, h*w)

                strfs /= (seq_len - tau) # tau, (h*w)
                strfs = to_np(strfs.reshape((-1, h, w)))

                all_units_strfs[j] = strfs
            _plot_strf(all_units_strfs, tau, result_path, hidden_size)

    param_path = os.path.join(result_path, 'hyperparameters.txt')
    with open(param_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

if __name__ == "__main__":
    start_time = time.time()
    main(args)
    print(f'Completed, total time: {time.time() - start_time}')
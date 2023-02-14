import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import random
import numpy as np
import math

def get_mnist(datapath, sample_size, sample_size_test, batch_size, seed, device, binary=False, classes=None):
    # classes: a list of specific class to sample from
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.MNIST(datapath, train=True, transform=transform, download=True)
    test = datasets.MNIST(datapath, train=False, transform=transform, download=True)

    # subsetting data based on sample size and number of classes
    idx = sum(train.targets == c for c in classes).bool() if classes else range(len(train))
    train.targets = train.targets[idx]
    train.data = train.data[idx]
    if sample_size != len(train):
        random.seed(seed)
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))
    random.seed(seed)
    test = torch.utils.data.Subset(test, random.sample(range(len(test)), sample_size_test))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X, y = [], []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
        y.append(targ)
    X = torch.cat(X, dim=0).to(device) # size, 28, 28
    y = torch.cat(y, dim=0).to(device)

    X_test, y_test = [], []
    for batch_idx, (data, targ) in enumerate(test_loader):
        X_test.append(data)
        y_test.append(targ)
    X_test = torch.cat(X_test, dim=0).to(device) # size, 28, 28
    y_test = torch.cat(y_test, dim=0).to(device)

    if binary:
        X[X > 0.5] = 1
        X[X < 0.5] = 0
        X_test[X_test > 0.5] = 1
        X_test[X_test < 0.5] = 0

    print(X.shape)
    return (X, y), (X_test, y_test)


def get_rotating_mnist(datapath, seq_len, seed, device, angle=np.pi/5, digit=0, test_digit=1):
    """digit: digit used to train the model
    
    test_digit: digit used to test the generalization of the model

    angle: rotating angle at each step
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.MNIST(datapath, train=True, transform=transform, download=True)

    dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # get data from particular classes
    idx = (train.targets == digit).bool()
    test_idx = (train.targets == test_digit).bool()
    train_data = train.data[idx] / 255.
    test_data = train.data[test_idx] / 255.

    # sample 1 image from train and test
    random.seed(seed)
    rdn_idx_train = random.randint(0, train_data.shape[0])
    rdn_idx_test = random.randint(0, test_data.shape[0])
    train_data = train_data[rdn_idx_train:rdn_idx_train+1].unsqueeze(1).to(device)
    test_data = test_data[rdn_idx_test:rdn_idx_test+1].unsqueeze(1).to(device)

    # rotate images
    train_sequence = []
    test_sequence = []

    for l in range(seq_len):
        theta = torch.tensor(angle * l)
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                [torch.sin(theta), torch.cos(theta), 0]])
        rot_mat = rot_mat[None, ...].repeat(train_data.shape[0], 1, 1).type(dtype)
        grid = F.affine_grid(rot_mat, train_data.size(), align_corners=False).type(dtype)
        train_sequence.append(F.grid_sample(train_data.type(dtype), grid, align_corners=False))
        test_sequence.append(F.grid_sample(test_data.type(dtype), grid, align_corners=False))
    
    return torch.cat(train_sequence, dim=0), torch.cat(test_sequence, dim=0)


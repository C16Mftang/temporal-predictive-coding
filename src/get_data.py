import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import math
import os

class DataWrapper(Dataset):
    """
    Class to wrap a dataset. Assumes X and y are already
    torch tensors and have the right data type and shape.
    
    Parameters
    ----------
    X : torch.Tensor
        Features tensor.
    y : torch.Tensor
        Labels tensor.
    """
    def __init__(self, X):
        self.features = X
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], []

def get_nat_movie(datapath, train_size):
    """
    Function to load Seb's natural movie data

    Inputs:
        datapath: specifies the directory containing the npy file

        train_size: number of movies used for training

        thresh: the threshold determining the dynamical level of the movies
    """
    d_path = os.path.join(datapath, 'nat_dynamic.npy')
    movie = np.load(d_path, mmap_mode='r+')
    print(movie.shape)
    train = movie[:train_size]
    
    return train

def get_moving_blobs(movie_num, frame_num, h, w):
    """Function to generate moving Gaussian blobs"""
    def _gaussian_blob(x, y, x0, y0, sigma_x, sigma_y, rho):
        inv_cov = np.linalg.inv([[sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y**2]])
        a = inv_cov[0, 0]
        b = inv_cov[0, 1]
        c = inv_cov[1, 1]
        z = a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2
        return np.exp(-0.5 * z)

    def _initialize_frame(h, w, sigma_x, sigma_y, rho):
        x0, y0 = np.random.randint(0, w), np.random.randint(0, h)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        frame = _gaussian_blob(x, y, x0, y0, sigma_x, sigma_y, rho)
        angle = np.random.uniform(0, 2 * np.pi)
        return frame, (x0, y0), angle

    def _move_blob(frame, center, angle, velocity, sigma_x, sigma_y, rho):
        h, w = frame.shape
        dx = int(velocity * np.cos(angle))
        dy = int(velocity * np.sin(angle))
        new_center = (center[0] + dx, center[1] + dy)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        new_frame = _gaussian_blob(x, y, new_center[0], new_center[1], sigma_x, sigma_y, rho)
        return new_frame, new_center

    movies = np.zeros((movie_num, frame_num, h, w))
    velocity = 1.5 # fix the velocity for all movies

    for i in range(movie_num):
        # Randomize sigma_x, sigma_y, and rho for each movie
        sigma_x = np.random.uniform(1.0, 3.0)
        sigma_y = np.random.uniform(1.0, 3.0)
        rho = np.random.uniform(-0.9, 0.9)  # Keeping rho in this range to ensure the covariance matrix is positive definite

        frame, center, angle = _initialize_frame(h, w, sigma_x, sigma_y, rho)
        movies[i, 0] = frame
        for j in range(1, frame_num):
            frame, center = _move_blob(frame, center, angle, velocity, sigma_x, sigma_y, rho)
            movies[i, j] = frame

    return movies


def get_seq_mnist(datapath, seq_len, sample_size, batch_size, seed, device):
    """Get batches of sequence mnist
    
    The data should be of shape [sample_size, seq_len, h, w]
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.MNIST(datapath, train=True, transform=transform, download=True)
    # test = datasets.MNIST(datapath, train=False, transform=transform, download=True)

    # each sample is a sequence of randomly sampled mnist digits
    # we could thus sample samplesize x seq_len images
    random.seed(seed)
    train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size * seq_len))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size * seq_len, shuffle=False)

    return train_loader


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


def get_rotating_mnist(datapath, 
                       seq_len, 
                       sample_size, 
                       test_size, 
                       batch_size, 
                       seed, 
                       device, 
                       angle=10, 
                       test_digit=9):
    """digit: digit used to train the model
    
    test_digit: digit used to test the generalization of the model

    angle: rotating angle at each step
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.MNIST(datapath, train=True, transform=transform, download=True)

    # randomly sample 
    dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # get data from particular classes
    idx = (train.targets != test_digit).bool()
    test_idx = (train.targets == test_digit).bool()
    train_data = train.data[idx] / 255.
    test_data = train.data[test_idx] / 255.

    random.seed(seed)
    train_data = train_data[random.sample(range(len(train_data)), sample_size)] # [sample_size, h, w]
    test_data = test_data[random.sample(range(len(test_data)), test_size)]
    h, w = train_data.shape[-2], train_data.shape[-1]
    # rotate images
    train_sequences = torch.zeros((sample_size, seq_len, h, w))

    for l in range(seq_len):
        train_sequences[:, l] = TF.rotate(train_data, angle * l)

    train_loader = DataLoader(DataWrapper(train_sequences), batch_size=batch_size)
    
    return train_loader, test_data


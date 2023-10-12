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

def get_nat_movie(datapath, train_size, seq_len):
    """
    Function to load Seb's natural movie data

    Inputs:
        datapath: specifies the directory containing the npy file

        train_size: number of movies used for training

        thresh: the threshold determining the dynamical level of the movies
    """
    d_path = os.path.join(datapath, 'nat_high_dynamic.npy')
    movie = np.load(d_path, mmap_mode='r+')
    train = movie[:train_size, :seq_len]
    if seq_len <= 20:
        train2 = movie[:train_size, seq_len:seq_len*2]
        train = np.concatenate((train, train2), axis=0) 
        print('Train size changed, now 2 * train_size')
    print(train.shape)
    
    return train

def get_moving_blobs(movie_num, frame_num, h, w, velocity, offset_factor=1):
    """Function to generate moving Gaussian blobs"""
    # def _gaussian_blob(x, y, x0, y0, sigma_x, sigma_y, rho):
    #     inv_cov = np.linalg.inv([[sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y**2]])
    #     a = inv_cov[0, 0]
    #     b = inv_cov[0, 1]
    #     c = inv_cov[1, 1]
    #     z = a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2
    #     return np.exp(-0.5 * z)

    # def _initialize_frame(h, w, sigma_x, sigma_y, rho):
    #     x0, y0 = np.random.randint(0, w), np.random.randint(0, h)
    #     x, y = np.meshgrid(np.arange(w), np.arange(h))
    #     frame = _gaussian_blob(x, y, x0, y0, sigma_x, sigma_y, rho)
    #     angle = np.random.uniform(0, 2 * np.pi)
    #     return frame, (x0, y0), angle

    # def _move_blob(frame, center, angle, velocity, sigma_x, sigma_y, rho):
    #     h, w = frame.shape
    #     dx = int(velocity * np.cos(angle))
    #     dy = int(velocity * np.sin(angle))
    #     new_center = (center[0] + dx, center[1] + dy)
        
    #     # Check for collisions with the borders and reflect the movement if necessary
    #     if new_center[0] <= 0 or new_center[0] >= w:
    #         angle = np.pi - angle
    #     if new_center[1] <= 0 or new_center[1] >= h:
    #         angle = -angle

    #     # Recalculate movement after reflection
    #     dx = int(velocity * np.cos(angle))
    #     dy = int(velocity * np.sin(angle))
    #     new_center = (center[0] + dx, center[1] + dy)

    #     x, y = np.meshgrid(np.arange(w), np.arange(h))
    #     new_frame = _gaussian_blob(x, y, new_center[0], new_center[1], sigma_x, sigma_y, rho)
    #     return new_frame, new_center, angle  # Return the updated angle

    # movies = np.zeros((movie_num, frame_num, h, w))
    # # velocity = 1.5 # fix the velocity for all movies

    # for i in range(movie_num):
    #     sigma_x = np.random.uniform(1.0, 3.0)
    #     sigma_y = np.random.uniform(1.0, 3.0)
    #     rho = np.random.uniform(-0.9, 0.9)

    #     frame, center, angle = _initialize_frame(h, w, sigma_x, sigma_y, rho)
    #     movies[i, 0] = frame
    #     for j in range(1, frame_num):
    #         frame, center, angle = _move_blob(frame, center, angle, velocity, sigma_x, sigma_y, rho)
    #         movies[i, j] = frame

    # return movies
    def _gaussian_blob(x, y, x0, y0, sigma_x, sigma_y, rho):
        inv_cov = np.linalg.inv([[sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y**2]])
        a = inv_cov[0, 0]
        b = inv_cov[0, 1]
        c = inv_cov[1, 1]
        z = a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2
        return np.exp(-0.5 * z)

    def _negative_blob(x, y, x0, y0, sigma_x, sigma_y, rho):
        return -_gaussian_blob(x, y, x0, y0, sigma_x, sigma_y, rho)


    def _initialize_frame(h, w, sigma_x, sigma_y, rho):
        # Define a margin, say 25% of the width and height
        margin_w = int(0.25 * w)
        margin_h = int(0.25 * h)
        
        # Adjust the random range to be within the central area
        x0 = np.random.randint(margin_w, w - margin_w)
        y0 = np.random.randint(margin_h, h - margin_h)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        frame = _gaussian_blob(x, y, x0, y0, sigma_x, sigma_y, rho)
        angle = np.random.uniform(0, 2*np.pi)
        
        offset = offset_factor * sigma_x  # Distance to place the side blobs, adjust as needed

        # Offset the x and y coordinates for the side blobs
        dx = int(offset * np.sin(angle))
        dy = int(offset * np.cos(angle))

        frame += _negative_blob(x, y, x0 + dx, y0 + dy, sigma_x, sigma_y, rho)
        frame += _negative_blob(x, y, x0 - dx, y0 - dy, sigma_x, sigma_y, rho)
        
        return frame, (x0, y0), angle


    def _move_blob(frame, center, angle, velocity, sigma_x, sigma_y, rho):
        h, w = frame.shape
        dx = int(velocity * np.cos(angle))
        dy = int(velocity * np.sin(angle))
        new_center = (center[0] + dx, center[1] + dy)
        
        # # Check for collisions with the borders and reflect the movement if necessary
        # if new_center[0] <= 0 or new_center[0] >= w:
        #     angle = np.pi - angle
        # if new_center[1] <= 0 or new_center[1] >= h:
        #     angle = -angle

        # # Recalculate movement after reflection
        # dx = int(velocity * np.cos(angle))
        # dy = int(velocity * np.sin(angle))
        # new_center = (center[0] + dx, center[1] + dy)

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        new_frame = _gaussian_blob(x, y, new_center[0], new_center[1], sigma_x, sigma_y, rho)

        # Determine side blobs' positions, same as above
        offset = offset_factor * sigma_x
        dx = int(offset * np.sin(angle))
        dy = int(offset * np.cos(angle))
        
        new_frame += _negative_blob(x, y, new_center[0] + dx, new_center[1] + dy, sigma_x, sigma_y, rho)
        new_frame += _negative_blob(x, y, new_center[0] - dx, new_center[1] - dy, sigma_x, sigma_y, rho)
        
        return new_frame, new_center, angle

    movies = np.zeros((movie_num, frame_num, h, w))

    for i in range(movie_num):
        sigma_x = np.random.uniform(2, 3)
        sigma_y = sigma_x * np.random.uniform(0.5, 0.6)
        rho_magnitude = np.random.uniform(0.9, 0.95)
        # rho_magnitude = 0.9
        rho_sign = np.random.choice([-1, 1])
        rho = rho_magnitude * rho_sign

        frame, center, angle = _initialize_frame(h, w, sigma_x, sigma_y, rho)
        movies[i, 0] = frame
        for j in range(1, frame_num):
            frame, center, angle = _move_blob(frame, center, angle, velocity, sigma_x, sigma_y, rho)
            movies[i, j] = frame
    movies = movies + np.random.normal(0, 0.1, movies.shape)
    return movies

def get_moving_bars(movie_num, frame_num, h, w, bar_width=2):
    # Initialize the output array
    movies = np.zeros((movie_num, frame_num, h, w), dtype=np.float32)

    for movie_idx in range(movie_num):
        # randiomly select velocity
        velocity = 1
        # Randomly decide if the bar is horizontal or vertical
        is_horizontal = np.random.choice([True, False])

        if is_horizontal:
            bar_pos = np.random.randint(0, h-bar_width)  # Initial position of the bar
            direction = 1 if np.random.rand() > 0.5 else -1  # Up or down
        else:
            bar_pos = np.random.randint(0, w-bar_width)  # Initial position of the bar
            direction = 1 if np.random.rand() > 0.5 else -1  # Left or right

        for frame_idx in range(frame_num):
            if is_horizontal:
                movies[movie_idx, frame_idx, bar_pos:bar_pos+bar_width, :] = 1.0
                bar_pos += direction * velocity

                # Bounce back if the bar hits the border
                if bar_pos <= 0 or bar_pos >= h-bar_width:
                    direction = -direction
                    bar_pos += direction * velocity
            else:
                movies[movie_idx, frame_idx, :, bar_pos:bar_pos+bar_width] = 1.0
                bar_pos += direction * velocity

                # Bounce back if the bar hits the border
                if bar_pos <= 0 or bar_pos >= w-bar_width:
                    direction = -direction
                    bar_pos += direction * velocity

    return movies

def get_bar_patches(sample_size, seq_len, h, w, simple=False):
    # Variables
    frame_size = 100
    num_lines = 50
    
    # Initialize the movie with gray background
    movie = np.zeros((seq_len, frame_size, frame_size))
    np.random.seed(48)
    
    # Define the lines
    directions = ['horizontal'] if simple else ['horizontal', 'vertical']
    movements = [1] if simple else [-1, 1]
    lines = [{'position': np.random.randint(0, frame_size),
              'color': np.random.choice([-1, 1]),
              'direction': np.random.choice(directions),
              'movement': np.random.choice(movements)} for _ in range(num_lines)]
    
    # Draw the lines in the first frame
    for line in lines:
        if line['direction'] == 'horizontal':
            movie[0, line['position'], :] = line['color']
        else:
            movie[0, :, line['position']] = line['color']
    
    # For each subsequent frame, move the lines
    for t in range(1, seq_len):
        for line in lines:
            if line['direction'] == 'horizontal':
                new_position = line['position'] + line['movement']
                # Check for bouncing
                if new_position < 0 or new_position >= frame_size:
                    line['movement'] = 0 if simple else -line['movement']
                    new_position = line['position'] + line['movement']
                line['position'] = new_position
                movie[t, line['position'], :] = line['color']
            else:
                new_position = line['position'] + line['movement']
                # Check for bouncing
                if new_position < 0 or new_position >= frame_size:
                    line['movement'] = -line['movement']
                    new_position = line['position'] + line['movement']
                line['position'] = new_position
                movie[t, :, line['position']] = line['color']
    
    # Extract random patches
    dataset = np.zeros((sample_size, seq_len, h, w))
    for i in range(sample_size):
        x = np.random.randint(0, frame_size - w + 1)
        y = np.random.randint(0, frame_size - h + 1)
        dataset[i] = movie[:, x:x+w, y:y+h]
    
    return dataset


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


import numpy as np


def linear(x):
    return x


def linear_deriv(x):
    return np.array([np.sum(np.eye(x.shape[0]), axis=0)]).reshape(2, )


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - tanh(x) ** 2

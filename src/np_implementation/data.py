import numpy as np

from src.np_implementation.activation import tanh as f


def _generate_noise(mean=0., sd=0.1, shape=None):
    return np.random.normal(mean, sd, shape if shape is not None else 2)


def generate_random_nonlinear_data(sampling_frequency, n_dimension, n_samples, x, C, A, seed=0, dt=None):
    data = np.zeros((n_dimension, n_samples))
    if dt is None:
        dt = 1 / sampling_frequency

    np.random.seed(seed)

    for i in range(len(data[1])):
        noise_data = _generate_noise(mean=0., sd=0.1, shape=data[:, i].shape)
        noise_hidden = np.sqrt(dt) * _generate_noise(mean=0., sd=0.1, shape=data[:, i].shape)
        data[:, i] = ((C @ f(x))[..., np.newaxis] + noise_data[..., np.newaxis]).flatten()
        x = x + dt * A @ f(x) + noise_hidden.flatten()

    return data, A

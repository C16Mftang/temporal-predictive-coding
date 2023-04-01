import numpy as np

from tqdm import tqdm
from loguru import logger


class TPC:
    def __init__(self, input_array: np.ndarray, A: np.ndarray, C: np.ndarray, dt: float = 0.5, k1: float = 0.001,
                 k2: float = 0.008, activation: str = 'nonlinear'):
        self.input = input_array
        self.input_size = len(input_array[1])
        self.predicted_data = np.zeros((input_array.shape[0], input_array.shape[1]))
        self.A = A
        self.C = C
        self.dt = dt
        self.k1, self.k2 = k1, k2
        self.error = np.zeros((1, input_array.shape[1]))

        if activation == 'linear':
            from src.np_implementation.activation import linear as f
            from src.np_implementation.activation import linear_deriv as df
            self.f = f
            self.df = df
        elif activation == 'nonlinear':
            from src.np_implementation.activation import tanh as f
            from src.np_implementation.activation import tanh_deriv as df
            self.f = f
            self.df = df
        else:
            logger.error(f'Invalid activation: {activation}. Only "linear" or "nonlinear" allowed.')
            raise KeyError()

        logger.info(f'Temporal Predictive Coding using a {activation} function')

    def get_error(self) -> np.ndarray:
        return self.error.flatten()

    def get_learned_A(self) -> np.ndarray:
        return self.A

    def get_learned_C(self) -> np.ndarray:
        return self.C

    def get_predictions(self) -> np.ndarray:
        return self.predicted_data

    def forward(self, C_decay: int = None, A_decay: int = None):
        x = np.zeros(self.input.shape[0])

        if C_decay:
            C_decay_counter = 1
        if A_decay:
            A_decay_counter = 1

        for t in tqdm(range(self.input_size)):
            ox = x.copy()
            x = x + self.dt * self.A @ self.f(ox)
            self.predicted_data[:, t] = self.C @ self.f(ox)
            e_y = self.input[:, t] - self.predicted_data[:, t]
            x = x + self.dt * (self.C.T @ (self.df(x) * e_y))
            e_x = self.C.T @ self.df(x) * e_y
            self.C += self.dt * (self.k1 * e_y[..., np.newaxis] @ self.f(x)[..., np.newaxis].T)
            self.A += self.dt * (self.k2 * e_x[..., np.newaxis] @ self.f(ox)[..., np.newaxis].T)
            if A_decay:
                if A_decay_counter == A_decay:
                    self.k2 /= 1.015
                    C_decay_counter = 1
            if C_decay:
                if C_decay_counter == C_decay:
                    self.k1 /= 1.015
                    C_decay_counter = 1
            self.error[:, t] = (np.linalg.norm(
                self.input[:, t] - self.predicted_data[:, t]) ** 2) / len(self.input[:, t])
            if A_decay:
                A_decay_counter += 1
            if C_decay:
                C_decay_counter += 1

import argparse

import numpy as np

from src.np_implementation.data import generate_random_nonlinear_data
from src.np_implementation.model import TPC


class ProgramArguments(object):
    def __init__(self):
        self.activation = None
        self.timepoints = None
        self.sampling_freq = None


def main():
    args = parse_args()

    if args.activation is None:
        activation = 'nonlinear'
    else:
        activation = args.activation

    if args.timepoints is None:
        timepoints = 4200
    else:
        timepoints = args.timepoints

    if args.sampling_freq is None:
        sampling_freq = 2
    else:
        sampling_freq = 2

    dt = 1 / sampling_freq
    x = np.array((1., 0.))
    C_init = np.eye(2) * 2.5
    A_init = (np.array((
        (-dt / 2, 1.),
        (-1., -dt / 2)))) * 2.5

    # generate random data
    y_truth, A_truth = generate_random_nonlinear_data(2, 2, timepoints, x, C_init, A_init)

    # infer the data with temporal predictive coding
    A = np.zeros((y_truth.shape[0], y_truth.shape[0]))
    C = np.eye(y_truth.shape[0])
    tpc = TPC(y_truth, A, C, activation=activation)
    tpc.forward()
    error = tpc.get_error()

    print("Nonlinear Model Error: ")
    print(error)


def parse_args():
    parser = argparse.ArgumentParser(description="A sample script to use the NumPy implementation of the "
                                                 "Temporal Predictive Coding algorithm.")
    parser.add_argument("--activation", help="Activation function. Only 'linear' or 'nonlinear' arguments"
                                             "allowed. The nonlinear function is Tanh. Default: 'nonlinear'. ")
    parser.add_argument("--timepoints", help="Number of time points to simulate. Default: 4200.")
    parser.add_argument("--sampling_freq", help="Sampling frequency to generate the data. Default: 2")

    program_arguments = ProgramArguments()
    parser.parse_args(namespace=program_arguments)

    return program_arguments


if __name__ == '__main__':
    main()

"""
Script to produce the plots in Figure 6 in the paper.
"""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

from src.np_implementation.model import TPC

plt.style.use('ggplot')

results_path = Path('./results/', 'pendulum')
results_path.mkdir(parents=True, exist_ok=True)


def pendulum_equation(t, theta):
    g = 9.81  # gravity acceleration (m/s^2)
    L = 3  # length of pendulum (m)
    b = 0.0  # damping factor (kg/s)
    m = 0.1  # mass (kg)

    dtheta2_dt = -(b / m * theta[1]) - (g / L * np.sin(theta[0]))
    dtheta1_dt = theta[1]

    return [dtheta1_dt, dtheta2_dt]


def simulate(integration_step, et):
    # initial and end values:
    st = 0  # start time (s)
    # et = 2500.4            # end time (s)
    ts = integration_step  # time step (s)

    theta1_init = 1.8  # initial angular displacement (rad)
    theta2_init = 2.2  # initial angular velocity (rad/s)
    theta_init = [theta1_init, theta2_init]
    t_span = [st, et + st]
    t = np.arange(st, et + st, ts)
    sim_points = len(t)
    l = np.arange(0, sim_points, 1)

    theta12 = solve_ivp(pendulum_equation, t_span, theta_init, t_eval=t)

    return theta12, t


if __name__ == '__main__':
    step = dt = 0.1
    et = 2500.4  # end time (s)
    n_simulations = 100
    nonlinear_errors = []
    linear_errors = []

    for i in range(n_simulations):
        ground_truth, time = simulate(step, et)
        np.random.seed(i)
        ground_truth.y[0, :] += np.random.normal(0, 0.1, ground_truth.y[0].shape)  # * step
        ground_truth.y[1, :] += np.random.normal(0, 0.1, ground_truth.y[0].shape)  # * step

        sol = np.zeros((ground_truth.y.shape[0], ground_truth.y.shape[1]))
        sol[0, :] = ground_truth.y[1, :]
        sol[1, :] = ground_truth.y[0, :]

        theta1 = ground_truth.y[0, :]  # angular displacement
        theta2 = ground_truth.y[1, :]  # angular velocity

        # use nonlinear model
        A = np.zeros((ground_truth.y.shape[0], ground_truth.y.shape[0]))
        C = np.eye(ground_truth.y.shape[0])
        tpc = TPC(ground_truth.y, A, C, activation='nonlinear', dt=0.1, k1=8.5, k2=0.9)
        tpc.forward(C_decay=300)
        e_nl = tpc.get_error()
        nonlinear_errors.append(np.mean(e_nl))
        A_pred_nl = tpc.get_learned_A()
        data_pred_nl = tpc.get_predictions()
        a_nl = tpc.get_learned_A()
        pred_sol_nl = np.zeros((ground_truth.y.shape[0], ground_truth.y.shape[1]))
        pred_sol_nl[0, :] = data_pred_nl[1, :]
        pred_sol_nl[1, :] = data_pred_nl[0, :]

        # use linear model
        A = np.zeros((ground_truth.y.shape[0], ground_truth.y.shape[0]))
        C = np.eye(ground_truth.y.shape[0])
        tpc = TPC(ground_truth.y, A, C, activation='linear', dt=0.1, k1=0.01, k2=0.09)
        tpc.forward(A_decay=500)
        e_l = tpc.get_error()
        linear_errors.append(np.mean(e_l))
        A_pred_l = tpc.get_learned_A()
        data_pred_l = tpc.get_predictions()
        a_l = tpc.get_learned_A()
        pred_sol_l = np.zeros((ground_truth.y.shape[0], ground_truth.y.shape[1]))
        pred_sol_l[0, :] = data_pred_l[1, :]
        pred_sol_l[1, :] = data_pred_l[0, :]

    nl_error_mean = np.mean(nonlinear_errors, axis=0)
    nl_error_sd = np.std(nonlinear_errors, axis=0)
    nl_error_se = np.std(nonlinear_errors, axis=0, ddof=1) / np.sqrt(np.size(nonlinear_errors, axis=0))

    l_error_mean = np.mean(linear_errors, axis=0)
    l_error_sd = np.std(linear_errors, axis=0)
    l_error_se = np.std(linear_errors, axis=0, ddof=1) / np.sqrt(np.size(linear_errors, axis=0))

    labels = ["Linear", "Nonlinear"]
    plt.figure(figsize=(6, 6))
    barlist = plt.bar(labels, [l_error_mean, nl_error_mean], 0.5,
                      yerr=[l_error_se, nl_error_se],
                      error_kw=dict(lw=5, capsize=5, capthick=3))
    barlist[0].set_color('#13678A')
    plt.xticks(color='k')
    plt.yticks(color='k')
    plt.title('Average mean-squared prediction errors')
    plt.ylabel('Average MSE', color='k')
    plt.legend(prop={'size': 9}, ncol=1)
    plt.tight_layout()
    plt.savefig(results_path / 'a_barplots.pdf')
    plt.show()

    fn = 80
    ff = sol[0].shape[0] - fn
    X, Y = np.mgrid[(-np.pi):np.pi:-30j, -4:4:30j]
    st = 0  # start time (s)
    ts = step  # time step (s)
    theta1_init = 1.8  # initial angular displacement (rad)
    theta2_init = 2.2  # initial angular velocity (rad/s)
    theta_init = [theta1_init, theta2_init]
    t_span = [st, et + st]
    U, V = pendulum_equation(t_span, [X, Y])
    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, U, V, color='purple')
    plt.plot(sol[1, ff:], sol[0, ff:], 'k-', linewidth=3, label='True')
    plt.plot(data_pred_nl[0, ff:], data_pred_nl[1, ff:], linewidth=3, label='Nonlinear model')
    plt.plot(data_pred_l[0, ff:], data_pred_l[1, ff:], '#13678A', linewidth=3, label='Linear model')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.title('Mean phase portrait')
    plt.xlabel(r'$\theta_1$', fontsize=20, color='k')
    plt.ylabel(r'$\theta_2$', fontsize=20, color='k')
    plt.legend(prop={'size': 9}, ncol=1)
    plt.tight_layout()
    plt.savefig(results_path / 'phase_portrait.pdf')
    plt.show()

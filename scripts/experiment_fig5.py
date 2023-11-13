"""
Script to produce the plots in Figure 5 in the paper.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import *

from src.np_implementation.data import generate_random_nonlinear_data
from src.np_implementation.model import TPC

plt.style.use('ggplot')

results_path = Path('./results/', 'nonlinear_linear_comparisons')
results_path.mkdir(parents=True, exist_ok=True)

PLOT_ESTIMATES = True
PLOT_ERRORS = True
PLOT_A_ERRORS = True
PLOT_A_MATRICES = True
PLOT_PARAM_SIN = True


def __moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def _annotated_heatmap(input_data, std_err, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    sns.heatmap(input_data, ax=ax, **kwargs)

    # Annotate the heatmap with mean values and standard errors
    for i in range(input_data.shape[0]):
        for j in range(input_data.shape[1]):
            ax.text(j + 0.5, i + 0.5, f"{input_data[i, j]:.4f} \n± {std_err[i, j]:.4f}",
                    ha="center", va="center", fontsize=24,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
    return ax


if __name__ == '__main__':
    # some parameters used for this experiment
    n_simulations = 100
    time_points = 5500

    # generate random data
    dt = 1 / 2
    x = np.array((1., 0.))
    C_init = np.eye(2) * 2.5
    A_init = (np.array((
        (-dt / 2, 1.),
        (-1., -dt / 2)))) * 2.5
    y_truth, A_truth = generate_random_nonlinear_data(2, 2, time_points, x, C_init, A_init)

    # define some variables to help us with the visualisations
    y_truth_full = np.zeros((n_simulations, 1, time_points))
    data_pred_nl_full = np.zeros((n_simulations, 1, time_points))
    data_pred_l_full = np.zeros((n_simulations, 1, time_points))

    errors_nonlinear = np.zeros((n_simulations, time_points))
    errors_linear = np.zeros((n_simulations, time_points))

    errors_A_nonlinear = np.zeros((n_simulations, 1))
    errors_A_linear = np.zeros((n_simulations, 1))

    As_nonlinear = np.zeros((n_simulations, 2, 2))
    As_linear = np.zeros((n_simulations, 2, 2))

    # run the simulations (by changing the `seed`, we change the random data)
    for i in range(n_simulations):
        y_truth, A_truth = generate_random_nonlinear_data(2, 2, time_points, x, C_init, A_init, seed=i)
        y_truth_full[i] = y_truth[0]

        # run nonlinear model
        A = np.zeros((y_truth.shape[0], y_truth.shape[0]))
        C = np.eye(y_truth.shape[0])
        tpc = TPC(y_truth, A, C, activation='nonlinear')
        tpc.forward(A_decay=500)
        e_nl = tpc.get_error()
        A_pred_nl = tpc.get_learned_A()
        data_pred_nl = tpc.get_predictions()
        a_nl = tpc.get_learned_A()

        data_pred_nl_full[i] = data_pred_nl[0]  # add all predictions into one big array
        errors_nonlinear[i] = e_nl  # add all errors into one big array

        mse_nl = ((A_truth - A_pred_nl) ** 2).mean()  # mean-squared error of learned A and true A
        errors_A_nonlinear[i] = mse_nl

        As_nonlinear[i] = a_nl

        # run linear model
        A = np.zeros((y_truth.shape[0], y_truth.shape[0]))
        C = np.eye(y_truth.shape[0])
        tpc = TPC(y_truth, A, C, activation='linear')
        tpc.forward(A_decay=500)
        e_l = tpc.get_error()
        A_pred_l = tpc.get_learned_A()
        data_pred_l = tpc.get_predictions()
        a_l = tpc.get_learned_A()

        data_pred_l_full[i] = data_pred_l[0]  # add all predictions into one big array
        errors_linear[i] = e_l  # add all errors into one big array

        mse_l = ((A_truth - A_pred_l) ** 2).mean()  # mean-squared error of learned A and true A
        errors_A_linear[i] = mse_l

        As_linear[i] = a_l

    solution = np.zeros((y_truth.T.shape[0], 3))
    solution[:, 0] = y_truth.T[:, 0]
    solution[:, 1] = data_pred_nl.T[:, 0]
    solution[:, 2] = data_pred_l.T[:, 0]

    if PLOT_ESTIMATES:
        ###
        # Simulation plots
        ###
        plt.figure(figsize=(6, 6))
        plt.plot(solution[:, 0].T, c='k', label='True')
        plt.plot(solution[:, 1].T, label='Nonlinear model')
        plt.plot(solution[:, 2].T, c='#13678A', label='Linear model')
        plt.xlim((0, 50))
        plt.xticks(np.arange(0, 60, 10), np.arange(0, 60, 10), color='k', fontsize=18)
        plt.yticks([-2.5, 2.5], color='k', fontsize=18)
        plt.title('True state vs. estimations', fontsize=24)
        plt.xlabel('Time', color='k', fontsize=18)
        plt.ylabel('Magnitude', color='k', fontsize=18)
        plt.legend(prop={'size': 20}, ncol=1)
        plt.tight_layout()
        plt.savefig(results_path / 'estimations_first_50.pdf')
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.plot(solution[:, 0].T, c='k', label='True')
        plt.plot(solution[:, 1].T, label='Nonlinear model')
        plt.plot(solution[:, 2].T, c='#13678A', label='Linear model')
        plt.xlim((time_points-51, time_points-1))
        plt.xticks(np.arange(time_points-51, time_points+9, 10),
                   np.arange(time_points-51, time_points+9, 10),
                   color='k', fontsize=18)
        plt.yticks([-2.5, 2.5], color='k', fontsize=18)
        plt.title('True state vs. estimations', fontsize=24)
        plt.xlabel('Time', color='k', fontsize=18)
        plt.ylabel('Magnitude', color='k', fontsize=18)
        plt.legend(prop={'size': 20}, ncol=1)
        plt.tight_layout()
        plt.savefig(results_path / 'estimations_last_50.pdf')
        plt.show()

    if PLOT_ERRORS:
        moving = 100

        nl_error_mean = np.mean(errors_nonlinear, axis=0)
        nl_error_se = np.std(errors_nonlinear, axis=0, ddof=1) / np.sqrt(np.size(errors_nonlinear, axis=0))

        plt.figure(figsize=(10, 6))
        plt.plot(__moving_average(nl_error_mean, moving), label='Nonlinear model', linewidth=3)
        plt.fill_between(range(len(__moving_average(nl_error_mean, moving))),
                         __moving_average(nl_error_mean, moving) - __moving_average(nl_error_se, moving),
                         __moving_average(nl_error_mean, moving) + __moving_average(nl_error_se, moving), alpha=0.45)

        l_error_mean = np.mean(errors_linear, axis=0)
        l_error_se = np.std(errors_linear, axis=0, ddof=1) / np.sqrt(np.size(errors_linear, axis=0))

        plt.plot(__moving_average(l_error_mean, moving), c='#13678A', label='Linear model', linewidth=3)
        plt.fill_between(range(len(__moving_average(l_error_mean, moving))),
                         __moving_average(l_error_mean, moving) - __moving_average(l_error_se, moving),
                         __moving_average(l_error_mean, moving) + __moving_average(l_error_se, moving), alpha=0.45,
                         color='#13678A')

        plt.xlim((0, len(__moving_average(nl_error_mean, moving))))
        plt.ylim((0))
        plt.xticks(np.arange(0, time_points+9, 1000),
                   np.arange(0, time_points+9, 1000),
                   color='k', fontsize=18)
        plt.yticks(color='k', fontsize=18)
        plt.title('Mean & standard errors of loss', fontsize=24)
        plt.xlabel('Time', color='k', fontsize=18)
        plt.ylabel('Error', color='k', fontsize=18)
        plt.legend(prop={'size': 20}, ncol=1)
        plt.tight_layout()
        plt.savefig(results_path / 'loss.pdf')
        plt.show()

    if PLOT_A_ERRORS:
        ###
        # Learned A matrices errors bar plot
        ###
        labels = ["Linear", "Nonlinear"]
        plt.figure(figsize=(6, 6))
        barlist = plt.bar(labels, [errors_A_linear.mean(), errors_A_nonlinear.mean()], 0.5,
                          yerr=[np.std(errors_A_linear), np.std(errors_A_nonlinear)],
                          error_kw=dict(lw=5, capsize=5, capthick=3))
        barlist[0].set_color('#13678A')
        plt.xticks(color='k', fontsize=18)
        plt.yticks(color='k', fontsize=18)
        plt.title('Average MSE of learned A', fontsize=24)
        plt.ylabel('MSE', color='k', fontsize=18)
        plt.legend(prop={'size': 20}, ncol=1)
        plt.tight_layout()
        plt.savefig(results_path / 'a_barplots.pdf')
        plt.show()

    if PLOT_A_MATRICES:
        import seaborn as sns

        # # True A
        # _, ax = plt.subplots(figsize=(6, 6))

        # sns.heatmap(A_truth, ax=ax, linewidth=.5, cmap="cividis", cbar=False,
        #             vmin=-2.5, vmax=2.5)

        # for i in range(A_truth.shape[0]):
        #     for j in range(A_truth.shape[1]):
        #         ax.text(j + 0.5, i + 0.5, f"{A_truth[i, j]:.3f}",
        #                 ha="center", va="center", fontsize=24,
        #                 bbox=dict(facecolor="white", edgecolor="none", pad=0.2))

        # plt.title('True A', fontsize=24)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(results_path / 'a_true.pdf')
        # plt.show()

        # # Nonlinear A
        # plt.figure(figsize=(6, 6))
        nl_A_sd = np.std(As_nonlinear, axis=0)
        nl_A_se = np.std(As_nonlinear, axis=0, ddof=1) / np.sqrt(np.size(As_nonlinear, axis=0))
        # _annotated_heatmap(np.mean(As_nonlinear, axis=0), nl_A_se, linewidth=.5, cmap="cividis", cbar=False,
        #                    vmin=-2.5, vmax=2.5)
        # plt.title('Learned A (nonlinear)', fontsize=24)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(results_path / 'a_nonlinear.pdf')
        # plt.show()

        # # Linear A
        # plt.figure(figsize=(6, 6))
        l_A_sd = np.std(As_linear, axis=0)
        l_A_se = np.std(As_linear, axis=0, ddof=1) / np.sqrt(np.size(As_linear, axis=0))
        # _annotated_heatmap(np.mean(As_linear, axis=0), l_A_se, linewidth=.5, cmap="cividis", cbar=False,
        #                    vmin=-2.5, vmax=2.5)

        # plt.title('Learned A (linear)', fontsize=24)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(results_path / 'a_linear.pdf')
        # plt.show()

        # do everything in one plot
        fig, ax = plt.subplots(1, 3, figsize=(8, 3))
        As = [A_truth, np.mean(As_nonlinear, axis=0), np.mean(As_linear, axis=0)]
        for i, a in enumerate(ax.flatten()):
            im = a.imshow(As[i], cmap="Spectral", vmin=-2.5, vmax=2.5)
            a.set_title(['True A', 'Learned A (nonlinear)', 'Learned A (linear)'][i])
            a.axis('off')
        # plot colorbar, make sure the colorbar is the same size as the plot
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=18) 
        plt.savefig(results_path / 'a_all.pdf', bbox_inches='tight')
        plt.show()

    if PLOT_PARAM_SIN:
        # to plot a parametric sinusoidal data with coefficients and constants optimised to match the ground-truth.
        from scipy.optimize import leastsq

        # fitting
        data = solution[:, 0]
        data = data[time_points-51:time_points-1].T
        N = 50  # number of data points
        t = np.linspace(0, 4 * np.pi, N)
        guess_mean = np.mean(data)
        guess_std = 3 * np.std(data) / (2 ** 0.5) / (2 ** 0.5)
        guess_phase = 1
        guess_freq = 2
        guess_amp = 1
        optimize_func = lambda x: x[0] * np.sin(x[1] * t + x[2]) - data
        est_amp, est_freq, est_phase = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase])[0]
        data_fit = est_amp * np.sin(est_freq * t + est_phase)
        sol = np.zeros((data.shape[0], 2))
        sol[:, 0] = data
        sol[:, 1] = data_fit

        # plot
        plt.figure(figsize=(6, 6))
        plt.plot(sol[:, 0].T, c='k', label='True')
        plt.plot(sol[:, 1].T, c='#FF00FF', label=r'$a \times sin(b \times t + c)$')
        plt.xlim((0, 50))
        plt.xticks(np.arange(0, 60, 10), np.arange(0, 60, 10), color='k', fontsize=18)
        plt.yticks([-3, 3], color='k', fontsize=18)
        plt.title('True state vs. fitted sin model', fontsize=24)
        plt.xlabel('Time', color='k', fontsize=18)
        plt.ylabel('Magnitude', color='k', fontsize=18)
        plt.legend(prop={'size': 20}, ncol=1, loc='upper right')
        plt.tight_layout()
        plt.savefig(results_path / 'sin_true_fit.pdf')
        plt.show()

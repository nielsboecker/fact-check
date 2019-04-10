import argparse

import matplotlib.pyplot as plt

from _4_B_fit_LR_model import LOSS_HISTORY_FREQUENCY
from dataaccess.files_io import read_pickle
from util.plots import show_plot_and_save_figure, prepare_seaborn_plots

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='use less data and less learning iterations', action='store_true')
args = parser.parse_args()


def plot_multiple_loss_values(learning_rates: list, multiple_loss_values: list):
    prepare_seaborn_plots()

    plt.xlabel('Iterations')
    plt.ylabel('Cross-Entropy Loss')

    # Fixed n for the given trained models
    plt.figtext(0.68, 0.56, r'$n = {:,}$'.format(100000))

    for i, values in enumerate(multiple_loss_values):
        label = r'$\alpha = {:,}$'.format(float(learning_rates[i]))
        x_axis = [i * LOSS_HISTORY_FREQUENCY for i in range(len(values))]
        plt.plot(x_axis, values, linewidth=2, label=label)

    plt.legend(loc='upper right')
    show_plot_and_save_figure('logistic_regression_loss_values_comparision.png')


if __name__ == '__main__':
    # load the pre-computed values
    learning_rates = ['0.0001', '0.001', '0.01', '0.1', '1.0']
    filepaths = ['./generated/logistic_regression_loss_{}.p'.format(rate) for rate in learning_rates]
    loss_values = [read_pickle(path) for path in filepaths]

    plot_multiple_loss_values(learning_rates, loss_values)

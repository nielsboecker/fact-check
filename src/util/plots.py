from time import gmtime, strftime

import seaborn as sns
from matplotlib import pyplot as plt

from dataaccess.files_constants import GENERATED_FIGURES_BASE_PATH
from dataaccess.files_io import create_dir_if_not_exists


def show_plot_and_save_figure(figure_name: str):
    time = strftime("%Y%m%d_%H%M%S", gmtime())
    figure_path = '{}{}_{}'.format(GENERATED_FIGURES_BASE_PATH, time, figure_name)
    create_dir_if_not_exists(figure_path)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()


def prepare_seaborn_plots():
    sns.set()
    sns.set_style('darkgrid', {'font.family': ['serif'], 'font.serif': 'Liberation'})
    sns.set_palette('husl')


def plot_loss_values(num_iterations: int, learning_rate: float, loss_values: list, step: int):
    prepare_seaborn_plots()
    plt.xlabel('Iterations')
    plt.ylabel('Cross-Entropy Loss')
    plt.figtext(0.65, 0.8, r'$n = {:,}$'.format(num_iterations))
    plt.figtext(0.65, 0.75, r'$\alpha = {:,}$'.format(learning_rate))
    plt.plot([i * step for i in range(len(loss_values))], loss_values, linewidth=4)
    show_plot_and_save_figure('logistic_regression_loss_values.png')
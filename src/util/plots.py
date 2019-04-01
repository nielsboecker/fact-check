from time import gmtime, strftime

import matplotlib.pyplot as plt
import seaborn as sns

from dataaccess.constants import GENERATED_FIGURES_BASE_PATH
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

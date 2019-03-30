import matplotlib.pyplot as plt
import seaborn as sns

from dataaccess.constants import GENERATED_BASE_PATH


def show_plot_and_save_figure(figure_name: str):

    figure_path = GENERATED_BASE_PATH + figure_name
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()


def prepare_seaborn_plots():
    sns.set()
    sns.set_style('darkgrid', {'font.family': ['serif'], 'font.serif': 'Liberation'})
    sns.set_palette('husl')
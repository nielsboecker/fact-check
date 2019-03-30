import matplotlib.pyplot as plt
import seaborn as sns

from dataaccess.constants import GENERATED_BASE_PATH


def show_plot_and_save_figure(figure_name: str):
    sns.set()

    figure_path = GENERATED_BASE_PATH + figure_name
    plt.savefig(figure_path, dpi=300)
    plt.show()

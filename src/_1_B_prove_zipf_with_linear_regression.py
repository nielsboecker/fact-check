import math

import matplotlib.pyplot as plt
from scipy import stats

from dataaccess.constants import GENERATED_COUNTS_PATH
from dataaccess.files_io import read_jsonl_and_map_to_df
from util.plots import show_plot_and_save_figure, prepare_seaborn_plots


def verify_plot_zipfs_law():
    counts = read_jsonl_and_map_to_df(GENERATED_COUNTS_PATH)

    x_ranks = range(1, len(counts[1]) + 1)
    y_counts = [count for count in counts[1]]
    x_ranks_log = [math.log10(rank) for rank in x_ranks]
    y_counts_log = [math.log10(count) for count in y_counts]

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_ranks_log, y_counts_log)
    r_squared = r_value ** 2
    print('slope: {}; intercept: {}; r-squared: {}'.format(slope, intercept, r_squared))

    prepare_seaborn_plots()
    plt.plot(x_ranks_log, y_counts_log, 'o')  # label='word frequencies')
    plt.plot(x_ranks_log, [intercept + slope * rank for rank in x_ranks_log], 'red')  # , label='fitted line')

    plt.xlabel('$log(rank)$')
    plt.ylabel('$log(frequency)$')
    plt.figtext(0.3, 0.45, '$R^2 = {:.5f}$'.format(r_squared))
    # plt.legend()

    show_plot_and_save_figure('1_distribution_zipf.png')


if __name__ == '__main__':
    verify_plot_zipfs_law()

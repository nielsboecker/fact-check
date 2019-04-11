import math

import matplotlib.pyplot as plt
from scipy import stats

from dataaccess.files_constants import GENERATED_COUNTS_PATH
from dataaccess.files_io import read_jsonl_and_map_to_df
from util.plots import show_plot_and_save_figure, prepare_seaborn_plots

if __name__ == '__main__':
    counts = read_jsonl_and_map_to_df(GENERATED_COUNTS_PATH)

    x_ranks = range(1, len(counts[1]) + 1)
    y_counts = [count for count in counts[1]]
    x_ranks_log = [math.log10(rank) for rank in x_ranks]
    y_counts_log = [math.log10(count) for count in y_counts]

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_ranks_log, y_counts_log)
    r_squared = r_value ** 2
    print('slope: {}; intercept: {}; r-squared: {}, p: {}'.format(slope, intercept, r_squared, p_value))

    prepare_seaborn_plots()
    plt.plot(x_ranks_log, y_counts_log, 'o')
    plt.plot(x_ranks_log, [intercept + slope * rank for rank in x_ranks_log], 'red')

    plt.xlabel('log(rank)')
    plt.ylabel('log(frequency)')
    plt.figtext(0.2, 0.45, 'R$^2$ = {:.5f}'.format(r_squared))
    #plt.figtext(0.2, 0.4, 'p < 4E-324')
    # plt.figtext(0.2, 0.35, 'standard error = {:.3E}'.format(Decimal(std_err)))

    show_plot_and_save_figure('1_distribution_zipf.png')

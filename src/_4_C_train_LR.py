import argparse

import pandas as pd

from _4_B_fit_LR_model import fit_and_get_model, LOSS_HISTORY_FREQUENCY
from dataaccess.files_constants import GENERATED_LOGISTIC_REGRESSION_MODEL, \
    GENERATED_LOGISTIC_REGRESSION_LOSS_HISTORY, GENERATED_NN_PREPROCESSED_TRAINING_DATA
from dataaccess.files_io import read_pickle, write_pickle
from util.LR_NN_preprocessing import extract_input_and_expected
from util.plots import plot_loss_values

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='use less data and less learning iterations', action='store_true')
parser.add_argument('--num_iterations', type=int, default=100000)
parser.add_argument('--learning_rate', type=float, default=0.1)
args = parser.parse_args()


if __name__ == '__main__':
    training_data: pd.DataFrame = read_pickle(GENERATED_NN_PREPROCESSED_TRAINING_DATA)
    train_input, train_expected = extract_input_and_expected(training_data)

    model, loss_values = fit_and_get_model(train_input, train_expected, args.num_iterations, args.learning_rate)
    write_pickle(GENERATED_LOGISTIC_REGRESSION_LOSS_HISTORY, loss_values)  # for plotting
    write_pickle(GENERATED_LOGISTIC_REGRESSION_MODEL, model)

    plot_loss_values(args.num_iterations, args.learning_rate, loss_values, LOSS_HISTORY_FREQUENCY)

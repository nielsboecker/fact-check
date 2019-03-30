import numpy as np
import pandas as pd

from _4_B_logistic_regression import fit_and_get_model
from dataaccess.constants import GENERATED_PREPROCESSED_TRAINING_DATA, GENERATED_LOGISTIC_REGRESSION_MODEL
from dataaccess.files_io import read_pickle, write_pickle

#if __name__ == '__main__':
training_data : pd.DataFrame = read_pickle(GENERATED_PREPROCESSED_TRAINING_DATA)

# use DataFrame.values to unpack the ndarrays in each cell
train_input_values = training_data['input_vector'].values
train_input = np.array([line for line in  train_input_values])
train_expected = training_data['expected_output']

model = fit_and_get_model(train_input, train_expected)
write_pickle(GENERATED_LOGISTIC_REGRESSION_MODEL, model)

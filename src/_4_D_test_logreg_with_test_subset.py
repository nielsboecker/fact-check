import numpy as np

from _4_B_logistic_regression import LogisticRegressionModel
from dataaccess.constants import GENERATED_PREPROCESSED_DEV_DATA, \
    GENERATED_LOGISTIC_REGRESSION_MODEL
from dataaccess.files_io import read_pickle
from util.logreg_preprocessing import extract_input_and_expected


def get_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    correspondence = predicted == actual
    return correspondence.mean()


if __name__ == '__main__':
    model: LogisticRegressionModel = read_pickle(GENERATED_LOGISTIC_REGRESSION_MODEL)
    dev_data = read_pickle(GENERATED_PREPROCESSED_DEV_DATA)
    dev_input, dev_expected = extract_input_and_expected(dev_data)

    dev_predicted = model.predict(dev_input)
    accuracy = get_accuracy(dev_predicted, dev_expected)
    print(accuracy)
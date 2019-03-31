import numpy as np

from _4_B_logistic_regression import LogisticRegressionModel
from dataaccess.constants import GENERATED_LOGISTIC_REGRESSION_MODEL, GENERATED_PREPROCESSED_DEV_DATA
from dataaccess.files_io import read_pickle
from util.evaluation import get_true_positive, get_false_positive, get_false_negative, get_baserate_prediction
from util.logreg_preprocessing import extract_input_and_expected


def get_precision(predicted: np.ndarray, actual: np.ndarray) -> float:
    true_positive = get_true_positive(predicted, actual)
    false_positive = get_false_positive(predicted, actual)
    return true_positive / (true_positive + false_positive)


def get_recall(predicted: np.ndarray, actual: np.ndarray) -> float:
    true_positive = get_true_positive(predicted, actual)
    false_negative = get_false_negative(predicted, actual)
    return true_positive / (true_positive + false_negative)


def get_f1_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    precision = get_precision(predicted, actual)
    recall = get_recall(predicted, actual)
    return 2 * (precision * recall) / (precision + recall)


if __name__ == '__main__':
    model: LogisticRegressionModel = read_pickle(GENERATED_LOGISTIC_REGRESSION_MODEL)
    dev_data = read_pickle(GENERATED_PREPROCESSED_DEV_DATA)
    dev_input, dev_expected = extract_input_and_expected(dev_data)

    model_prediction = model.predict(dev_input)
    model_precision = get_precision(model_prediction, dev_expected)
    model_recall = get_recall(model_prediction, dev_expected)
    model_f1 = get_f1_score(model_prediction, dev_expected)

    baserate_prediction = get_baserate_prediction(model_prediction)
    baserate_precision = get_precision(baserate_prediction, dev_expected)
    baserate_recall = get_recall(baserate_prediction, dev_expected)
    baserate_f1 = get_f1_score(baserate_prediction, dev_expected)

    print('*' * 60)
    print('Trained model\nPrecision: {:.5f}\nRecall: {:.5f}\nF1-Score: {:.5f}'.format(
        model_precision, model_recall, model_f1))
    print('*' * 60)
    print('Base-rate model\nPrecision: {:.5f}\nRecall: {:.5f}\nF1-Score: {:.5f}'.format(
        baserate_precision, baserate_recall, baserate_f1))

import argparse

from dataaccess.files_constants import GENERATED_LOGISTIC_REGRESSION_MODEL, GENERATED_LR_PREPROCESSED_DEV_DATA
from dataaccess.files_io import read_pickle
from model.logistic_regression import LogisticRegressionModel
from util.LR_NN_preprocessing import extract_input_and_expected
from util.evaluation import get_accuracy, get_baserate_predictions

parser = argparse.ArgumentParser()
parser.add_argument("--bias_corrected", help="use bias correction (King and Zeng, 2001)", action="store_true")
args = parser.parse_args()


if __name__ == '__main__':
    model: LogisticRegressionModel = read_pickle(GENERATED_LOGISTIC_REGRESSION_MODEL)
    dev_data = read_pickle(GENERATED_LR_PREPROCESSED_DEV_DATA)
    dev_input, dev_expected = extract_input_and_expected(dev_data)

    prediction_threshold = .5
    model_prediction = model.get_predictions(dev_input, prediction_threshold, args.bias_corrected)
    model_accuracy = get_accuracy(model_prediction, dev_expected)

    baserate_zeros_prediction = get_baserate_predictions(model_prediction)
    baserate_zeros_accuracy = get_accuracy(baserate_zeros_prediction, dev_expected)

    baserate_random_prediction = get_baserate_predictions(model_prediction, zeros=False)
    baserate_random_accuracy = get_accuracy(baserate_random_prediction, dev_expected)

    print('Trained model (threshold = {}) accuracy: {:.5f}'.format(prediction_threshold, model_accuracy))
    print('Base rate (zeros) accuracy: {:.5f}'.format(baserate_zeros_accuracy))
    print('Base rate (random) accuracy: {:.5f}'.format(baserate_random_accuracy))

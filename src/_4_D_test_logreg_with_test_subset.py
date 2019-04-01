from model.logistic_regression import LogisticRegressionModel
from dataaccess.files_constants import GENERATED_PREPROCESSED_DEV_DATA, \
    GENERATED_LOGISTIC_REGRESSION_MODEL
from dataaccess.files_io import read_pickle
from util.evaluation import get_accuracy, get_baserate_predictions
from util.logreg_preprocessing import extract_input_and_expected


if __name__ == '__main__':
    model: LogisticRegressionModel = read_pickle(GENERATED_LOGISTIC_REGRESSION_MODEL)
    dev_data = read_pickle(GENERATED_PREPROCESSED_DEV_DATA)
    dev_input, dev_expected = extract_input_and_expected(dev_data)

    prediction_threshold = .5
    model_prediction = model.get_predictions(dev_input, threshold=prediction_threshold)
    model_accuracy = get_accuracy(model_prediction, dev_expected)

    baserate_zeros_prediction = get_baserate_predictions(model_prediction)
    baserate_zeros_accuracy = get_accuracy(baserate_zeros_prediction, dev_expected)

    baserate_random_prediction = get_baserate_predictions(model_prediction, zeros=False)
    baserate_random_accuracy = get_accuracy(baserate_random_prediction, dev_expected)

    print('Trained model (threshold = {}) accuracy: {:.5f}'.format(prediction_threshold, model_accuracy))
    print('Base rate (zeros) accuracy: {:.5f}'.format(baserate_zeros_accuracy))
    print('Base rate (random) accuracy: {:.5f}'.format(baserate_random_accuracy))

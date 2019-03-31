from _4_B_logistic_regression import LogisticRegressionModel
from dataaccess.constants import GENERATED_PREPROCESSED_DEV_DATA, \
    GENERATED_LOGISTIC_REGRESSION_MODEL
from dataaccess.files_io import read_pickle
from util.evaluation import get_accuracy, get_baserate_prediction
from util.logreg_preprocessing import extract_input_and_expected


if __name__ == '__main__':
    model: LogisticRegressionModel = read_pickle(GENERATED_LOGISTIC_REGRESSION_MODEL)
    dev_data = read_pickle(GENERATED_PREPROCESSED_DEV_DATA)
    dev_input, dev_expected = extract_input_and_expected(dev_data)

    model_prediction = model.predict(dev_input)
    model_accuracy = get_accuracy(model_prediction, dev_expected)

    baserate_prediction = get_baserate_prediction(model_prediction)
    baserate_accuracy = get_accuracy(baserate_prediction, dev_expected)

    print('Trained model accuracy: {:.5f}'.format(model_accuracy))
    print('Base-rate accuracy: {:.5f}'.format(baserate_accuracy))

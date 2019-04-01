# Note: As this part of the analysis exceeds what is expected in the
# problem statements, some existing libraries are used for convenience

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score

from _4_B_logistic_regression import LogisticRegressionModel
from dataaccess.constants import GENERATED_LOGISTIC_REGRESSION_MODEL, GENERATED_PREPROCESSED_DEV_DATA
from dataaccess.files_io import read_pickle
from util.evaluation import get_baserate_predictions, \
    get_baserate_probabilities
from util.logreg_preprocessing import extract_input_and_expected
from util.plots import prepare_seaborn_plots, show_plot_and_save_figure


def plot_precision_recall_curve():
    prepare_seaborn_plots()
    # precision-recall curve for the model
    plt.plot(model_recall, model_precision, marker='.', label='Trained logistic regression')
    # precision-recall curve for the baserate
    plt.plot(baserate_recall, baserate_precision, marker='.', label='Base rate model')
    # no skill, i.e. percentage of 1's in the expected output
    baserate = np.count_nonzero(dev_expected) / len(dev_expected)
    plt.plot([0, 1], [baserate, baserate], linestyle='--', label='Percentage of events in data')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    show_plot_and_save_figure('precision_recall_curve.png')


if __name__ == '__main__':
    model: LogisticRegressionModel = read_pickle(GENERATED_LOGISTIC_REGRESSION_MODEL)
    dev_data = read_pickle(GENERATED_PREPROCESSED_DEV_DATA)
    dev_input, dev_expected = extract_input_and_expected(dev_data)

    model_prediction = model.get_predictions(dev_input)
    baserate_prediction = get_baserate_predictions(model_prediction)

    # predictions
    print('Evaluation preditions (int 0|1)')
    print('Predictions Baserate AUC: {:.2f}'.format(roc_auc_score(dev_expected, baserate_prediction)))
    print('Predictions Model AUC: {:.2f}'.format(roc_auc_score(dev_expected, model_prediction)))
    print('*' * 40)

    # probabilities
    print('Evaluation probabilites (float 0...1)')
    baserate_probabilities = get_baserate_probabilities(model_prediction)
    baserate_probabilities_auc = roc_auc_score(dev_expected, baserate_probabilities)
    print('Probabilities Baserate AUC: {:.2f}'.format(baserate_probabilities_auc))
    model_probabilities = model.get_probabilities(dev_input)
    model_probabilities_auc = roc_auc_score(dev_expected, model_probabilities)
    print('Probabilities Model AUC: {:.2f}'.format(model_probabilities_auc))
    print('*' * 40)

    # precision-recall curve
    baserate_precision, baserate_recall, baserate_thresholds = precision_recall_curve(dev_expected, baserate_probabilities)
    model_precision, model_recall, model_thresholds = precision_recall_curve(dev_expected, model_probabilities)

    # precision-recall AUC
    precision_recall_auc = auc(model_recall, model_precision)
    print('Precision-Recall AUC: {}'.format(precision_recall_auc))

    # average precision score
    avg_precision_score = average_precision_score(dev_expected, model_probabilities)
    print('Average Precision Score: {}'.format(avg_precision_score))

    plot_precision_recall_curve()

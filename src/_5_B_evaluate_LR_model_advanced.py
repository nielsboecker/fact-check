# Note: As this part of the analysis exceeds what is expected in the
# problem statements, some existing libraries are used for convenience
import argparse

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score, roc_curve

from dataaccess.files_constants import GENERATED_LOGISTIC_REGRESSION_MODEL, GENERATED_LR_PREPROCESSED_DEV_DATA
from dataaccess.files_io import read_pickle
from model.logistic_regression import LogisticRegressionModel
from util.evaluation import get_baserate_predictions, \
    get_baserate_probabilities
from util.logreg_preprocessing import extract_input_and_expected
from util.plots import prepare_seaborn_plots, show_plot_and_save_figure

parser = argparse.ArgumentParser()
parser.add_argument("--bias_corrected", help="use bias correction (King and Zeng, 2001)", action="store_true")
args = parser.parse_args()


def plot_precision_recall_curve():
    prepare_seaborn_plots()

    # precision-recall curve for the model and baserate
    plt.plot(model_recall, model_precision, marker='.', label='Trained logistic regression')
    plt.plot(baserate_recall, baserate_precision, marker='.', label='Base rate model')
    # percentage of 1's in the expected output
    percentage = np.count_nonzero(dev_expected) / len(dev_expected)
    plt.plot([0, 1], [percentage, percentage], linestyle='--', label='Percentage of events in data')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    show_plot_and_save_figure('precision_recall_curve.png')


def plot_roc_auc_curve():
    # ROC curve for the model and random base rate
    model_false_positives, model_true_positives, _ = roc_curve(dev_expected, model_probabilities)
    base_random_false_positives, base_random_true_positives, _ = roc_curve(dev_expected, baserate_probabilities)

    #
    zero_probabilites = get_baserate_predictions(dev_expected, zeros=True)
    base_zero_false_positives, base_zero_true_positives, _ = roc_curve(dev_expected, zero_probabilites)

    plt.plot(model_false_positives, model_true_positives, label='Trained logistic regression')
    plt.plot(base_zero_false_positives, base_zero_true_positives,label='Base rate model (zeros)')
    plt.plot(base_random_false_positives, base_random_true_positives,label='Base rate model (random)')
    # plt.plot([0, 1], [0, 1], linestyle='--', label='Random guessing expectation')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    show_plot_and_save_figure('roc_auc_curve')


if __name__ == '__main__':
    model: LogisticRegressionModel = read_pickle(GENERATED_LOGISTIC_REGRESSION_MODEL)
    dev_data = read_pickle(GENERATED_LR_PREPROCESSED_DEV_DATA)
    dev_input, dev_expected = extract_input_and_expected(dev_data)

    model_prediction = model.get_predictions(dev_input, args.bias_corrected)
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
    baserate_precision, baserate_recall, _ = precision_recall_curve(dev_expected, baserate_probabilities)
    model_precision, model_recall, _ = precision_recall_curve(dev_expected, model_probabilities)

    # precision-recall AUC
    precision_recall_auc = auc(model_recall, model_precision)
    print('Precision-Recall AUC: {}'.format(precision_recall_auc))

    # average precision score
    avg_precision_score = average_precision_score(dev_expected, model_probabilities)
    print('Average Precision Score: {}'.format(avg_precision_score))

    plot_roc_auc_curve()

    plot_precision_recall_curve()

import time

import numpy as np
from numpy import ndarray

from model.logistic_regression import LogisticRegressionModel, sigmoid, prepend_intercept

NUM_EPOCHS = 100000
LEARNING_RATE = 0.01
LOSS_HISTORY_FREQUENCY = 500


def get_loss(hypothesis: ndarray, actual: ndarray) -> float: # TODO
    return (-actual * np.log(hypothesis) - (1 - actual) * np.log(1 - hypothesis)).mean()


def fit_and_get_model(in_values: ndarray, out_expected: ndarray, debug: bool = False) -> tuple:
    global NUM_EPOCHS, LEARNING_RATE
    start_time = time.time()
    print('Start logistic regression fitting')

    in_values = prepend_intercept(in_values)

    dimension = in_values.shape[1]
    num_of_datapoints = len(out_expected) * dimension # out_expected.size

    # initialise weights as ndarray with length 601 (incl. intercept)
    weights = np.zeros(dimension)

    # keep track of loss values to verify learning
    loss_values = []

    if (debug):
        NUM_EPOCHS = 1000

    # iteratively improve weights to fit expected output better
    for i in range(NUM_EPOCHS):
        lin_regression = np.dot(in_values, weights)
        hypothesis = sigmoid(lin_regression)

        difference = hypothesis - out_expected
        gradient = np.dot(in_values.transpose(), difference) / num_of_datapoints
        weights -= LEARNING_RATE * gradient

        if i % LOSS_HISTORY_FREQUENCY == 0:
            loss_values.append(get_loss(hypothesis, out_expected))
        if i == NUM_EPOCHS - 1:
            print('Final loss: {}'.format(get_loss(hypothesis, out_expected)))

    print('Finished fitting after {:.2f} seconds'.format(time.time() - start_time))

    return LogisticRegressionModel(weights, NUM_EPOCHS, LEARNING_RATE), loss_values

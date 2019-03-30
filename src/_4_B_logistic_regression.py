import time

import numpy as np
from numpy import ndarray

NUM_EPOCHS = 10000       # TODO
LEARNING_RATE = 0.01     # TODO


class LogisticRegression:
    def __init__(self, weights: ndarray):
        self.weights = weights


    def predict(self, in_values: np.ndarray, threshold: float = 0.5) -> np.array:
        z = np.dot(in_values, self.weights)
        hypothesis = sigmoid(z)
        return hypothesis >= threshold


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def get_loss(hypothesis: ndarray, actual: ndarray) -> float: # TODO
    return (-actual * np.log(hypothesis) - (1 - actual) * np.log(1 - hypothesis)).mean()


def fit_and_get_model(in_values: ndarray, out_expected: ndarray) -> LogisticRegression:
    # TODO: intercept!
    start_time = time.time()

    dimension = in_values.shape[1]
    num_of_datapoints = len(out_expected) * dimension # out_expected.size

    # initialise weights as ndarray with length 600
    weights = np.zeros(dimension)

    #
    for i in range(NUM_EPOCHS):
        z = np.dot(in_values, weights)
        hypothesis = sigmoid(z)

        # TODO T vs transpose
        difference = hypothesis - out_expected
        gradient = np.dot(in_values.transpose(), difference) / num_of_datapoints
        weights -= LEARNING_RATE * gradient

        # TODO
        if i % 100 == 0:
            print(get_loss(hypothesis, out_expected))

    print('Finished fitting after {:.2f} seconds\nFinal weights: {}'.format(time.time() - start_time, weights))
    return LogisticRegression(weights)

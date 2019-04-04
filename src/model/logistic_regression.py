import numpy as np
from numpy.core.multiarray import ndarray


class LogisticRegressionModel:
    def __init__(self, weights: ndarray, num_epochs: int, learning_rate: float):
        self.weights = weights
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def get_probabilities(self, in_values: ndarray) -> ndarray:
        in_values = prepend_intercept(in_values)
        lin_regression = np.dot(in_values, self.weights)
        hypothesis = sigmoid(lin_regression)
        return hypothesis

    def get_predictions(self, in_values: ndarray, threshold: float = 0.5, bias_corrected: bool = False) -> ndarray:
        if bias_corrected:
            self.run_bias_prior_correction()
        return self.get_probabilities(in_values) >= threshold

    # refer to  gking.harvard.edu/files/gking/files/baby0s.pdf 
    def run_bias_prior_correction(self, population_probability: float = 10 / 14839, sample_probability: float = 0.5):
        factor_1 = (1 - population_probability) / population_probability
        factor_2 = sample_probability / (1 - sample_probability)
        correction = np.log(factor_1 * factor_2)
        self.weights -= correction



def relu(values: ndarray) -> ndarray:
    return np.maximum(values, 0)



def sigmoid(values: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-values))


def prepend_intercept(values: ndarray) -> ndarray:
    intercept = np.ones((values.shape[0], 1))
    return np.concatenate((intercept, values), axis=1)

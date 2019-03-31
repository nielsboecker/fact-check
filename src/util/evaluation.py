import numpy as np


def get_true_positive(predicted: np.ndarray, actual: np.ndarray) -> int:
    predicted_true = predicted == 1
    actual_true = actual == 1
    both_true = predicted_true == actual_true
    true_positives = np.count_nonzero(both_true)
    return true_positives


def get_false_positive(predicted: np.ndarray, actual: np.ndarray) -> int:
    predicted_true = predicted == 1
    actual_false = actual == 0
    false_prediction = predicted_true == actual_false
    false_positives = np.count_nonzero(false_prediction)
    return false_positives


def get_false_negative(predicted: np.ndarray, actual: np.ndarray) -> int:
    predicted_false = predicted == 0
    actual_true = actual == 1
    false_prediction = predicted_false == actual_true#
    false_negatives = np.count_nonzero(false_prediction)
    return false_negatives


def get_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    correspondence = predicted == actual
    return correspondence.mean()


def get_baserate_prediction(sample: np.ndarray) -> np.ndarray:
    return np.zeros_like(sample)

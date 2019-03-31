import numpy as np


def get_true_positive(predicted: np.ndarray, actual: np.ndarray) -> int:
    true_positives = 0
    for i in range (len(predicted)):
        if predicted[i] and actual[i]:
            true_positives += 1
    return true_positives


def get_false_positive(predicted: np.ndarray, actual: np.ndarray) -> int:
    false_positives = 0
    for i in range (len(predicted)):
        if predicted[i] and not actual[i]:
            false_positives += 1
    return false_positives


def get_false_negative(predicted: np.ndarray, actual: np.ndarray) -> int:
    false_negatives = 0
    for i in range (len(predicted)):
        if not predicted[i] and actual[i]:
            false_negatives += 1
    return false_negatives


def get_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    correspondence = predicted == actual
    return correspondence.mean()


def get_baserate_prediction(sample: np.ndarray) -> np.ndarray:
    return np.zeros_like(sample)

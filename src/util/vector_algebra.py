import numpy as np


def get_min_max_vectors(vectors: list) -> tuple:
    minimum = vectors[0]
    maximum = vectors[0]
    for vector in vectors:
        minimum = np.minimum(vector, minimum)
        maximum = np.maximum(vector, maximum)
    return minimum, maximum

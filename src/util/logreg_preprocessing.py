import numpy as np
import pandas as pd


def extract_input_and_expected(preprocessed: pd.DataFrame) -> tuple:
    input_values = preprocessed['input_vector'].values
    input = np.array([line for line in input_values])
    expected = preprocessed['expected_output']
    return input, expected

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

from dataaccess.files_io import read_pickle
from util.LR_NN_preprocessing import extract_input_and_expected


class FeverClaimsDataset(Dataset):
    def __init__(self, preprocessed_pickle_path: str):
        preprocessed_dataset: pd.DataFrame = read_pickle(preprocessed_pickle_path)
        inputs, labels = extract_input_and_expected(preprocessed_dataset)
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, index):
        input: torch.Tensor = torch.from_numpy(self.inputs[index]).float()
        label: int = self.labels[index]
        return input, label

    def __len__(self):
        return len(self.inputs)

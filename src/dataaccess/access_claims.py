import pandas as pd

from dataaccess.constants import DATA_TRAINING_PATH, CLAIMS_COLUMNS_LABELED, DATA_DEV_LABELED_PATH
from dataaccess.files_io import read_jsonl_and_map_to_df

claims_training = read_jsonl_and_map_to_df(DATA_TRAINING_PATH, CLAIMS_COLUMNS_LABELED).set_index('id', drop=False)
claims_dev = read_jsonl_and_map_to_df(DATA_DEV_LABELED_PATH, CLAIMS_COLUMNS_LABELED).set_index('id', drop=False)


def get_claim(id: int, dataset: str = 'train') -> str:
    return get_corresponding_dataset(dataset).loc[id]['claim']


def get_claim_row(id: int, dataset: str = 'train') -> pd.Series:
    return get_corresponding_dataset(dataset).loc[id]


def get_all_claims(dataset: str = 'train') -> pd.DataFrame:
    return get_corresponding_dataset(dataset)


def claim_is_verifiable(claim_id: int, dataset: str = 'train') -> bool:
    return get_corresponding_dataset(dataset).loc[claim_id]['verifiable'] == 'VERIFIABLE'


def get_corresponding_dataset(dataset: str) -> pd.DataFrame:
    if dataset.startswith('train'):
        return claims_training
    else:
        return claims_dev
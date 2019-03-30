import pandas as pd

from dataaccess.constants import DATA_TRAINING_PATH, CLAIMS_COLUMNS_LABELED
from dataaccess.files_io import read_jsonl_and_map_to_df

claims = read_jsonl_and_map_to_df(DATA_TRAINING_PATH, CLAIMS_COLUMNS_LABELED).set_index('id', drop=False)


def get_training_claim(id: int) -> str:
    return claims.loc[id]['claim']


def get_training_claim_row(id: int) -> pd.Series:
    return claims.loc[id]


def get_all_training_claims() -> pd.DataFrame:
    return claims


def training_claim_is_verifiable(claim_id: int) -> bool:
    return claims.loc[claim_id]['verifiable'] == 'VERIFIABLE'

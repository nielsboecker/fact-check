import pandas as pd

from dataaccess.constants import CLAIMS_COLUMNS_LABELED, DATA_DEV_LABELED_PATH
from dataaccess.files_io import read_jsonl_and_map_to_df

claims = read_jsonl_and_map_to_df(DATA_DEV_LABELED_PATH, CLAIMS_COLUMNS_LABELED).set_index('id', drop=False)


# def get_training_claim(id: int) -> str:
#     return claims.loc[id]['claim']
#

def get_dev_claim_row(id: int) -> pd.Series:
    return claims.loc[id]


def get_all_dev_claims() -> pd.DataFrame:
    return claims


def dev_claim_is_verifiable(claim_id: int) -> bool:
    return claims.loc[claim_id]['verifiable'] == 'VERIFIABLE'

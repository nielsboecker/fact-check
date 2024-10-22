import pandas as pd

from dataaccess.files_constants import GENERATED_COUNTS_PATH
from documentretrieval.data_constants import COLLECTION_TOTAL_WORDS
from dataaccess.files_io import read_jsonl_and_map_to_df

terms_with_occurrences = read_jsonl_and_map_to_df(GENERATED_COUNTS_PATH, ['term', 'occurrences']).set_index('term', drop=False)


def get_collection_occurrences_of_term(term: str) -> int:
    return terms_with_occurrences.loc[term]['occurrences']


def get_collection_probability_for_term(term: str) -> float:
    occurences = get_collection_occurrences_of_term(term)
    return occurences / COLLECTION_TOTAL_WORDS


def get_terms_with_occurrences_mapping() -> pd.DataFrame:
    return terms_with_occurrences

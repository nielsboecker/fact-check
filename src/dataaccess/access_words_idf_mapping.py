from dataaccess.constants import GENERATED_IDF_PATH
from dataaccess.json_io import read_jsonl_and_map_to_df

words_with_idf = read_jsonl_and_map_to_df(GENERATED_IDF_PATH, ['word', 'idf']).set_index('word', drop=False)


def get_idf_for_term(term: str) -> float:
    return words_with_idf.loc[term]['idf']

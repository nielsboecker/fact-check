from dataaccess.constants import GENERATED_IDF_PATH
from dataaccess.files_io import read_jsonl_and_map_to_df

words_with_idf = read_jsonl_and_map_to_df(GENERATED_IDF_PATH, ['word', 'idf']).set_index('word', drop=False)


def get_idf_for_term(term: str) -> float:
    try:
        return words_with_idf.loc[term]['idf']
    except KeyError:
        # this can happen for tokens from doc titles, as the IDF values are only generated for doc text
        return 0

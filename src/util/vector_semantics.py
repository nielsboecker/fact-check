import math
from collections import Counter

from dataaccess.constants import GENERATED_IDF_PATH
from dataaccess.json_io import read_jsonl_and_map_to_df

words_with_idf = read_jsonl_and_map_to_df(GENERATED_IDF_PATH, ['word', 'idf']).set_index('word', drop=False)


def get_tfidf_vector_norm(text: list, debug=False, variant='relative'):
    # Note: this takes the TF directly from the text argument and IDF from doc -> IDF mapping
    word_count = Counter(text)
    accumulated_tfidf_values_for_words = []
    for word, count in word_count.items():
        tf = count if variant == 'raw_count' else float(count) / len(text)
        idf = words_with_idf.loc[word]['idf']
        tfidf_value = tf * idf
        accumulated_tfidf_values_for_words.append(tfidf_value)
    norm = math.sqrt(sum([i ** 2 for i in accumulated_tfidf_values_for_words]))

    if (debug):
        print('Computed norm {} for doc \n{}'.format(norm, text))
    return norm

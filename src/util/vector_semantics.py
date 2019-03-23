import math
from collections import Counter

from documentretrieval.access_inverted_index import get_index_entry_for_term


def get_tfidf_vector_norm(text: list, debug=False, variant='relative'):
    # Note: this takes the TF directly from the text argument
    word_count = Counter(text)
    accumulated_tfidf_values_for_words = []
    for word, count in word_count.items():
        tf = count if variant == 'raw_count' else float(count) / len(text)
        index_entry = get_index_entry_for_term(word)
        idf = index_entry['idf']
        tfidf_value = tf * idf
        accumulated_tfidf_values_for_words.append(tfidf_value)
    norm = math.sqrt(sum([i ** 2 for i in accumulated_tfidf_values_for_words]))

    if (debug):
        print('Computed norm {} for doc \n{}'.format(norm, text))
    return norm

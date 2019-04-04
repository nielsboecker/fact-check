import math
from collections import Counter

from dataaccess.access_words_idf_mapping import get_idf_for_term


def get_tfidf_vector_norm(text: list, variant='relative'):
    # Note: this takes the TF directly from the text argument and IDF from doc -> IDF mapping
    word_count = Counter(text)
    accumulated_tfidf_values_for_words = []
    for word, count in word_count.items():
        tf = count if variant == 'raw_count' else float(count) / len(text)
        idf = get_idf_for_term(word)
        tfidf_value = tf * idf
        accumulated_tfidf_values_for_words.append(tfidf_value)
    norm = math.sqrt(sum([i ** 2 for i in accumulated_tfidf_values_for_words]))

    return norm

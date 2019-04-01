from dataaccess.access_terms_frequencies_mapping import get_terms_with_occurrences_mapping
from documentretrieval.data_constants import COLLECTION_TOTAL_WORDS

if __name__ == '__main__':
    vocabulary = get_terms_with_occurrences_mapping()
    vocabulary['frequency'] = vocabulary['occurrences'] / COLLECTION_TOTAL_WORDS
    # the data should already be sorted by rank anyway
    vocabulary['term_rank'] = vocabulary['occurrences'].rank(ascending=False)
    vocabulary['frequency_rank_product'] = vocabulary.apply(lambda row: row.frequency * row.term_rank, axis=1)

    mean_probability = vocabulary['frequency_rank_product'].mean()
    standard_deviation = vocabulary['frequency_rank_product'].std()
    print('Mean of frequency * rank = {:,}.\nStandard deviation = {:,}'.format(mean_probability, standard_deviation))

    print('Random samples from vocabulary')
    print(vocabulary.sample(n=10))

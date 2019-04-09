from dataaccess.access_terms_frequencies_mapping import get_terms_with_occurrences_mapping
from documentretrieval.data_constants import COLLECTION_TOTAL_WORDS

if __name__ == '__main__':
    vocabulary = get_terms_with_occurrences_mapping()
    vocabulary['probability'] = vocabulary['occurrences'] / COLLECTION_TOTAL_WORDS
    # the data should already be sorted by rank anyway
    vocabulary['term_rank'] = vocabulary['occurrences'].rank(ascending=False)
    vocabulary['probability_rank_product'] = vocabulary.apply(lambda row: row.probability * row.term_rank, axis=1)

    mean_probability = vocabulary['probability_rank_product'].mean()
    standard_deviation = vocabulary['probability_rank_product'].std()
    print('Mean of probability * rank = {:,}.\nStandard deviation = {:,}'.format(mean_probability, standard_deviation))

    print('Random samples from vocabulary')
    print(vocabulary.sample(n=10))

from functools import reduce
from operator import mul as multiply

from dataaccess.access_docs_lengths_mapping import get_length_of_doc
from dataaccess.constants import COLLECTION_FILTERED_VOCABULARY_SIZE


LINDSTONE_CORRECTION_FACTOR = 0.01


def get_query_likelihood_score_laplace_smoothing(claim_terms: list,
                                                 doc_with_coordination_terms: tuple,
                                                 lindstone_correction: float = None) -> tuple:
    page_id = doc_with_coordination_terms[0]
    coordination_terms_for_doc = doc_with_coordination_terms[1]
    doc_length = get_length_of_doc(page_id)

    epsilon = lindstone_correction or 1

    term_probabilities = []
    for term in claim_terms:
        if term in coordination_terms_for_doc.keys():
            occurrences = coordination_terms_for_doc[term]
            probability = (occurrences + epsilon) / (doc_length + epsilon * COLLECTION_FILTERED_VOCABULARY_SIZE)
            term_probabilities.append(probability)
        else:
            term_probabilities.append(0)

    query_likelihood = reduce(multiply, term_probabilities, 1)
    return page_id, query_likelihood


def get_query_likelihood_score_laplace_lindstone_smoothing(claim_terms: list,
                                                           doc_with_coordination_terms: tuple):
    return get_query_likelihood_score_laplace_smoothing(claim_terms, doc_with_coordination_terms, LINDSTONE_CORRECTION_FACTOR)

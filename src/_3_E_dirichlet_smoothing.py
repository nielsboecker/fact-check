from functools import reduce
from operator import mul as multiply

from dataaccess.access_docs_lengths_mapping import get_length_of_doc
from dataaccess.access_terms_frequencies_mapping import get_collection_probability_for_term
from dataaccess.constants import COLLECTION_TOTAL_WORDS, COLLECTION_DOCUMENTS_NUMBER

MU = COLLECTION_TOTAL_WORDS / COLLECTION_DOCUMENTS_NUMBER  # estimate of the average doc length, ~50 for these docs


def get_query_likelihood_score_dirichlet_smoothing(claim_terms: list,
                                                   doc_with_coordination_terms: tuple) -> tuple:
    page_id = doc_with_coordination_terms[0]
    coordination_terms_for_doc = doc_with_coordination_terms[1]
    doc_length = get_length_of_doc(page_id)

    term_probabilities = []
    for term in claim_terms:
        if term in coordination_terms_for_doc.keys():
            occurrences = coordination_terms_for_doc[term]
            document_probability = occurrences / doc_length
            collection_probability = get_collection_probability_for_term(term)

            lambda_ = doc_length / (doc_length + MU)
            probability = lambda_ * document_probability + (1 - lambda_) * collection_probability
            term_probabilities.append(probability)
        else:
            collection_probability = get_collection_probability_for_term(term)
            term_probabilities.append(collection_probability)

    query_likelihood = reduce(multiply, term_probabilities, 1)
    return page_id, query_likelihood

from functools import reduce
from operator import mul as multiply

from dataaccess.access_docs_lengths_mapping import get_length_of_doc


def get_query_likelihood_score_no_smoothing(claim_terms: list, doc_with_coordination_terms: tuple) -> tuple:
    page_id = doc_with_coordination_terms[0]
    coordination_terms_for_doc = doc_with_coordination_terms[1]
    doc_length = get_length_of_doc(page_id)

    # if any of the claim terms is missing from doc, we can short-circuit this computation
    if any([term not in coordination_terms_for_doc.keys() for term in claim_terms]):
        return page_id, 0

    term_probabilities = []
    for term in claim_terms:
        if term in coordination_terms_for_doc.keys():
            occurrences = coordination_terms_for_doc[term]
            probability = float(occurrences) / float(doc_length)
            term_probabilities.append(probability)
        else:
            # this should never run, because function would have returned already
            term_probabilities.append(0)

    query_likelihood = reduce(multiply, term_probabilities, 1)
    return page_id, query_likelihood

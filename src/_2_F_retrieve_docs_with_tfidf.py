import argparse
import time
from collections import Counter
from operator import itemgetter

from termcolor import colored

from dataaccess.access_claims import get_claim_row, get_all_claims
from dataaccess.access_docs_norms_mapping import get_norm_for_doc_text
from dataaccess.access_inverted_index import get_candidate_documents_for_claim
from dataaccess.access_words_idf_mapping import get_idf_for_term
from dataaccess.constants import RETRIEVED_TFIDF_DIRECTORY, \
    DOCS_TO_RETRIEVE_PER_CLAIM
from documentretrieval.claim_processing import preprocess_text, display_or_store_result
from documentretrieval.document_processing import preprocess_doc_title
from documentretrieval.term_processing import process_normalise_tokenise_filter
from util.theads_processes import get_process_pool
from util.vector_semantics import get_tfidf_vector_norm

parser = argparse.ArgumentParser()
parser.add_argument('--variant', help='TF weighting variant', choices=['raw_count', 'relative'], default='relative')
parser.add_argument('--doc_title', help='[0...1] weight of doc title vs. doc text', type=float, default=0.)
parser.add_argument('--dataset', choices=['train', 'dev'], type=str, default='train')
parser.add_argument('--id', help='process only this ID of a claim to retrieve for test purposes', type=int)
parser.add_argument('--limit', help='only use subset for the first 10 claims', action='store_true')
parser.add_argument('--print', help='print results rather than storing on disk', action='store_true')
parser.add_argument('--debug', help='show more print statements', action='store_true')
args = parser.parse_args()


def get_tfidf_vector_for_document(coordination_terms_for_doc: dict, claim_terms: list):
    # the vectors only need to have a dimension for each term that occurs in
    # the query, as other terms would be 0 in the dot product anyways
    unique_sorted_claim_terms = sorted(list(set(claim_terms)))
    doc_vector = [0 for _ in unique_sorted_claim_terms]

    # for terms that are in doc and claim, get IDF values as they are stored in the index
    for index, term in enumerate(unique_sorted_claim_terms):
        if term in coordination_terms_for_doc.keys():
            doc_vector[index] = coordination_terms_for_doc[term]

    return doc_vector


def get_tfidf_vector_for_claim(claim_terms: list):
    claim_terms_with_occurrences = Counter(claim_terms)
    claim_vector = []

    for term_with_count in sorted(claim_terms_with_occurrences.items()):
        term = term_with_count[0]
        occurrences = term_with_count[1]
        # For the claim, the occurrences determining the TF come directly from the claim
        tf = occurrences if args.variant == 'raw_count' else occurrences / len(claim_terms)
        idf = get_idf_for_term(term)
        tfidf = tf * idf
        claim_vector.append(tfidf)

    if args.debug:
        print('Computed vector {} for claim "{}"'.format(claim_vector, claim_terms))
    return claim_vector


def get_doc_product(vector1: list, vector2: list):
    assert len(vector1) == len(vector2)
    return sum([vector1[i] * vector2[i] for i in range(len(vector1))])


def get_claim_doc_text_cosine_similarity(claim_terms: list,
                                         claim_vector: list,
                                         claim_norm: float,
                                         doc_with_coordination_terms: tuple) -> float:
    page_id = doc_with_coordination_terms[0]
    coordination_terms_for_doc = doc_with_coordination_terms[1]
    doc_text_vector = get_tfidf_vector_for_document(coordination_terms_for_doc, claim_terms)
    # use pre-computed norm
    doc_text_norm = get_norm_for_doc_text(page_id)

    dot_product = get_doc_product(claim_vector, doc_text_vector)
    cosine_sim = dot_product / (claim_norm * doc_text_norm)

    return cosine_sim


def get_claim_doc_title_cosine_similarity(claim_terms: list,
                                          claim_vector: list,
                                          claim_norm: float,
                                          doc_with_coordination_terms: tuple) -> float:
    doc_title_terms = preprocess_doc_title(doc_with_coordination_terms[0])
    # Use only terms from title, discard doc text
    term_counts = dict(Counter(doc_title_terms))
    doc_title_vector = get_tfidf_vector_for_document(term_counts, claim_terms)
    doc_title_norm = get_tfidf_vector_norm(doc_title_terms, args.debug, args.variant)

    dot_product = get_doc_product(claim_vector, doc_title_vector)
    norms_product = claim_norm * doc_title_norm
    if not norms_product:
        # happens if all tokens in doc title are non-alphanumeric or unseen in IDF generation
        return 0

    cosine_sim = dot_product / norms_product
    return cosine_sim


def scoring_function(claim_terms: list,
                     claim_vector: list,
                     claim_norm: float,
                     doc_with_coordination_terms: tuple) -> tuple:
    page_id = doc_with_coordination_terms[0]
    cosine_claim_doc_text = get_claim_doc_text_cosine_similarity(claim_terms, claim_vector, claim_norm, doc_with_coordination_terms)
    cosine_claim_doc_title = get_claim_doc_title_cosine_similarity(claim_terms, claim_vector, claim_norm, doc_with_coordination_terms)
    title_weight = args.doc_title or 0
    score = title_weight * cosine_claim_doc_title + (1 - title_weight) * cosine_claim_doc_text
    # if args.debug:
    #     print('Claim: "{}", title: {}, score: {}'.format(claim_terms, doc_with_coordination_terms[0], score))
    return page_id, score


def retrieve_documents_for_claim(claim: str, claim_id: int):
    print(colored('Retrieving documents for claim [{}]: "{}"'.format(claim_id, claim), attrs=['bold']))
    preprocessed_claim = preprocess_text(claim)
    claim_terms = process_normalise_tokenise_filter(preprocessed_claim)
    claim_vector = get_tfidf_vector_for_claim(claim_terms)
    claim_norm = get_tfidf_vector_norm(claim_terms, args.debug, args.variant)

    # only docs that appear in index for at least one claim term to be considered
    doc_candidates = get_candidate_documents_for_claim(claim_terms)

    # similarity scores for each claim-doc combination
    docs_with_similarity_scores = [
        scoring_function(claim_terms, claim_vector, claim_norm, doc_with_terms) for doc_with_terms in
        doc_candidates.items()]

    # sort by similarity and limit to top results
    docs_with_similarity_scores.sort(key=itemgetter(1), reverse=True)
    result_docs = docs_with_similarity_scores[:DOCS_TO_RETRIEVE_PER_CLAIM]

    display_or_store_result(claim, claim_id, result_docs, RETRIEVED_TFIDF_DIRECTORY, args.print)


def retrieve_document_for_claim_row(claim_row: tuple):
    claim_id = claim_row[1]['id']
    claim = claim_row[1]['claim']
    retrieve_documents_for_claim(claim, claim_id)


def retrieve_documents_for_all_claims():
    claims = get_all_claims(dataset=args.dataset)

    pool = get_process_pool()
    if (args.limit):
        pool.map(retrieve_document_for_claim_row, claims.head(n=16).iterrows())
    else:
        pool.map(retrieve_document_for_claim_row, claims.iterrows())


if __name__ == '__main__':
    start_time = time.time()
    if args.id:
        claim = get_claim_row(args.id, dataset=args.dataset)
        document = retrieve_document_for_claim_row((None, claim))
    else:
        retrieve_documents_for_all_claims()
    print('Finished retrieval after {:.2f} seconds'.format(time.time() - start_time))

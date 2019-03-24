import argparse
import time
from collections import Counter
from operator import itemgetter

from termcolor import colored

from dataaccess.constants import GENERATED_IDF_PATH, DATA_TRAINING_PATH, RETRIEVED_TFIDF_DIRECTORY, \
    CLAIMS_COLUMNS_LABELED, GENERATED_DOCUMENT_NORMS_MAPPING, DOCS_TO_RETRIEVE_PER_CLAIM
from dataaccess.json_io import read_jsonl_and_map_to_df, write_list_to_jsonl
from documentretrieval.access_inverted_index import get_candidate_documents_for_claim
from documentretrieval.claim_processing import preprocess_claim
from documentretrieval.term_processing import process_normalise_tokenise_filter
from documentretrieval.wiki_page_retrieval import retrieve_wiki_page
from util.theads_processes import get_thread_pool
from util.vector_semantics import get_tfidf_vector_norm

parser = argparse.ArgumentParser()
parser.add_argument('--variant', help='TF weighting variant', choices=['raw_count', 'relative'], default='relative')
parser.add_argument('--id', help='ID of a claim to retrieve for test purposes (if defined, process only this one)', type=int)
parser.add_argument('--limit', help='only use subset for the first 10 claims', action='store_true')
parser.add_argument('--print', help='print results rather than storing on disk', action='store_true')
parser.add_argument('--debug', help='show more print statements', action='store_true')
args = parser.parse_args()

claims = read_jsonl_and_map_to_df(DATA_TRAINING_PATH, CLAIMS_COLUMNS_LABELED).set_index('id', drop=False)
words_with_idf = read_jsonl_and_map_to_df(GENERATED_IDF_PATH, ['word', 'idf']).set_index('word', drop=False)
docs_norms = read_jsonl_and_map_to_df(GENERATED_DOCUMENT_NORMS_MAPPING, ['doc', 'norm']).set_index('doc', drop=False)


def get_tfidf_vector_for_document(coordination_terms_for_doc: dict, claim_terms: list):
    # the vectors only need to have a dimension for each term that occurs in
    # the query, as other terms would be 0 in the dot product anyways
    unique_sorted_claim_terms = set(claim_terms)
    doc_vector = [0 for _ in unique_sorted_claim_terms]

    # for terms that are in doc and claim, get IDF values as they are stored in the index
    for index, term in enumerate(claim_terms):
        if term in coordination_terms_for_doc.keys():
            doc_vector[index] = coordination_terms_for_doc[term]

    return doc_vector


def get_tfidf_vector_for_claim(claim_terms: list):
    claim_terms_with_occurrences = Counter(claim_terms)
    claim_vector = [] #np.zeros(len(claim_terms_with_occurrences))

    for term_with_count in sorted(claim_terms_with_occurrences.items()):
        term = term_with_count[0]
        occurrences = term_with_count[1]
        # For the claim, the occurrences determining the TF come directly from the claim
        tf = occurrences if args.variant == 'raw_count' else occurrences / len(claim_terms)
        idf = words_with_idf.loc[term]['idf']
        tfidf = tf * idf
        claim_vector.append(tfidf)

    if (args.debug):
        print('Computed vector {} for claim "{}"'.format(claim_vector, claim_terms))
    return claim_vector


def get_doc_product(vector1: list, vector2: list):
    assert len(vector1) == len(vector2)
    return sum([vector1[i] * vector2[i] for i in range(len(vector1))])


def get_claim_doc_cosine_similarity(claim_terms: list, doc_with_coordination_terms: list) -> tuple:
    claim_vector = get_tfidf_vector_for_claim(claim_terms)
    claim_norm = get_tfidf_vector_norm(claim_terms, args.debug, args.variant)

    page_id = doc_with_coordination_terms[0]
    coordination_terms_for_doc = doc_with_coordination_terms[1]
    doc_vector = get_tfidf_vector_for_document(coordination_terms_for_doc, claim_terms)
    doc_norm = docs_norms.loc[page_id]['norm']

    dot_product = get_doc_product(claim_vector, doc_vector)
    cosine_sim = dot_product / (claim_norm * doc_norm)

    return (page_id, cosine_sim)


def retrieve_documents_for_claim(claim: str, claim_id: int):
    print(colored('Retrieving documents for claim [{}]: "{}"'.format(claim_id, claim), attrs=['bold']))
    preprocessed_claim = preprocess_claim(claim)
    claim_terms = process_normalise_tokenise_filter(preprocessed_claim)

    # only docs that appear in index for at least one claim term to be considered
    doc_candidates = get_candidate_documents_for_claim(claim_terms)

    # similarity scores for each claim-doc combination
    docs_with_similarity_scores = [get_claim_doc_cosine_similarity(claim_terms, doc_with_terms) for doc_with_terms in doc_candidates]

    # sort by similarity and limit to top results
    docs_with_similarity_scores.sort(key=itemgetter(1), reverse=True)
    result_docs = docs_with_similarity_scores[:DOCS_TO_RETRIEVE_PER_CLAIM]

    if (args.print):
        print(colored('Results for claim "{}":'.format(claim), attrs=['bold']))
        for doc in result_docs:
            page_id = doc[0]
            wiki_page = retrieve_wiki_page(page_id)
            print(wiki_page)
    else:
        result_path = '{}{}.jsonl'.format(RETRIEVED_TFIDF_DIRECTORY, claim_id)
        write_list_to_jsonl(result_path, result_docs)


def retrieve_document_for_claim_row(claim_row: tuple):
    claim_id = claim_row[1]['id']
    claim = claim_row[1]['claim']
    retrieve_documents_for_claim(claim, claim_id)


def retrieve_documents_for_all_claims():
    # TODO: thread vs process
    thread_pool = get_thread_pool()
    if (args.limit):
        thread_pool.map(retrieve_document_for_claim_row, claims.head(n=15).iterrows())
    else:
        thread_pool.map(retrieve_document_for_claim_row, claims.iterrows())


if __name__ == '__main__':
    start_time = time.time()
    if (args.id):
        claim = claims.loc[args.id]
        document = retrieve_document_for_claim_row((None, claim))
    else:
        retrieve_documents_for_all_claims()
    print('Finished retrieval after {:.2f} seconds'.format(time.time() - start_time))

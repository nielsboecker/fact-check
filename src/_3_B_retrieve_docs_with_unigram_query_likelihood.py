import argparse
import time
from operator import itemgetter

from termcolor import colored

from _3_C_probabilistic_no_smoothing import get_query_likelihood_score_no_smoothing
from dataaccess.access_inverted_index import get_candidate_documents_for_claim
from dataaccess.access_wiki_page import retrieve_wiki_page
from dataaccess.constants import DATA_TRAINING_PATH, CLAIMS_COLUMNS_LABELED, DOCS_TO_RETRIEVE_PER_CLAIM, \
    RETRIEVED_PROBABILISTIC_DIRECTORY
from dataaccess.json_io import read_jsonl_and_map_to_df, write_list_to_jsonl
from documentretrieval.claim_processing import preprocess_claim
from documentretrieval.term_processing import process_normalise_tokenise_filter
from util.theads_processes import get_thread_pool

parser = argparse.ArgumentParser()
parser.add_argument('--smoothing', type=str, choices=[None, 'laplace', 'jelinek_mercer', 'dirichlet'], default=None)
parser.add_argument('--remove_zero_likelihood', help='if documents yield query likelihood 0, don\' show them',
                    action='store_true')
parser.add_argument('--id', help='ID of a claim to retrieve for test purposes (if defined, process only this one)',
                    type=int)
parser.add_argument('--limit', help='only use subset for the first 10 claims', action='store_true')
parser.add_argument('--print', help='print results rather than storing on disk', action='store_true')
args = parser.parse_args()

claims = read_jsonl_and_map_to_df(DATA_TRAINING_PATH, CLAIMS_COLUMNS_LABELED).set_index('id', drop=False)


def retrieve_documents_for_claim(claim: str, claim_id: int):
    print(colored('Retrieving documents for claim [{}]: "{}"'.format(claim_id, claim), attrs=['bold']))
    preprocessed_claim = preprocess_claim(claim)
    claim_terms = process_normalise_tokenise_filter(preprocessed_claim)

    # only docs that appear in index for at least one claim term to be considered
    doc_candidates = get_candidate_documents_for_claim(claim_terms, mode='raw_count')

    retrieval_function = get_query_likelihood_score_no_smoothing
    if args.smoothing == 'laplace':
        retrieval_function = None
    if args.smoothing == 'jelinek_mercer':
        retrieval_function = None
    if args.smoothing == 'dirichlet':
        retrieval_function = None

    # query likelihood scores for each claim-doc combination
    docs_with_query_likelihood_scores = [retrieval_function(claim_terms, doc_with_terms) for
                                         doc_with_terms in
                                         doc_candidates.items()]

    # zero values lead to random retrievals if all documents evaluate to zero, so might rather want to show no results
    if (args.remove_zero_likelihood):
        docs_with_query_likelihood_scores = list(filter(lambda x: x[1] != 0, docs_with_query_likelihood_scores))

    # sort by query likelihood and limit to top results
    docs_with_query_likelihood_scores.sort(key=itemgetter(1), reverse=True)
    result_docs = docs_with_query_likelihood_scores[:DOCS_TO_RETRIEVE_PER_CLAIM]

    show_or_store_results(claim, claim_id, result_docs)


def show_or_store_results(claim, claim_id, result_docs):
    if (args.print):
        print(colored('Results for claim "{}":'.format(claim), attrs=['bold']))
        if not result_docs:
            print('-- No documents found --')
        for doc in result_docs:
            page_id = doc[0]
            wiki_page = retrieve_wiki_page(page_id)
            print(wiki_page)
    else:
        result_path = '{}{}.jsonl'.format(RETRIEVED_PROBABILISTIC_DIRECTORY, claim_id)
        write_list_to_jsonl(result_path, result_docs)


def retrieve_documents_for_claim_row(claim_row: tuple):
    claim_id = claim_row[1]['id']
    claim = claim_row[1]['claim']
    retrieve_documents_for_claim(claim, claim_id)


def retrieve_documents_for_all_claims():
    # TODO: thread vs process
    thread_pool = get_thread_pool()
    if (args.limit):
        thread_pool.map(retrieve_documents_for_claim_row, claims.head(n=15).iterrows())
    else:
        thread_pool.map(retrieve_documents_for_claim_row, claims.iterrows())


if __name__ == '__main__':
    start_time = time.time()
    if (args.id):
        claim = claims.loc[args.id]
        document = retrieve_documents_for_claim_row((None, claim))
    else:
        retrieve_documents_for_all_claims()
    print('Finished retrieval after {:.2f} seconds'.format(time.time() - start_time))

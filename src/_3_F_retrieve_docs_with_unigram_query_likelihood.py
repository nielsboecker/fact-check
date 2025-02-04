import argparse
import time
from operator import itemgetter

from termcolor import colored

from _3_B_probabilistic_no_smoothing import get_query_likelihood_score_no_smoothing
from _3_C_laplace_smoothing import get_query_likelihood_score_laplace_lindstone_smoothing, \
    get_query_likelihood_score_laplace_smoothing
from _3_D_jelinek_mercer_smoothing import get_query_likelihood_score_jelinek_mercer_smoothing
from _3_E_dirichlet_smoothing import get_query_likelihood_score_dirichlet_smoothing
from dataaccess.access_claims import get_all_claims, get_claim_row
from dataaccess.access_inverted_index import get_candidate_documents_for_claim
from dataaccess.files_constants import DOCS_TO_RETRIEVE_PER_CLAIM, \
    RETRIEVED_PROBABILISTIC_DIRECTORY
from documentretrieval.claim_processing import preprocess_claim_text, display_or_store_result
from documentretrieval.term_processing import process_normalise_tokenise_filter
from util.theads_processes import get_process_pool

parser = argparse.ArgumentParser()
parser.add_argument('--smoothing', type=str, default=None,
                    choices=[None, 'laplace', 'laplace_lindstone', 'jelinek_mercer', 'dirichlet'])
parser.add_argument('--remove_zero_likelihood', help='if documents yield query likelihood 0, don\' show them',
                    action='store_true')
parser.add_argument('--id', help='ID of a claim to retrieve for test purposes (if defined, process only this one)',
                    type=int)
parser.add_argument('--dataset', choices=['train', 'dev', 'test'], type=str, default='train')
parser.add_argument('--limit', help='only use subset for the first 10 claims', action='store_true')
parser.add_argument('--print', help='print results rather than storing on disk', action='store_true')
args = parser.parse_args()


def retrieve_documents_for_claim(claim: str, claim_id: int):
    print(colored('Retrieving documents for claim [{}]: "{}"'.format(claim_id, claim), attrs=['bold']))
    preprocessed_claim = preprocess_claim_text(claim)
    claim_terms = process_normalise_tokenise_filter(preprocessed_claim)

    # only docs that appear in index for at least one claim term to be considered
    doc_candidates = get_candidate_documents_for_claim(claim_terms, mode='raw_count')

    scoring_function = get_query_likelihood_score_no_smoothing
    if args.smoothing == 'laplace':
        scoring_function = get_query_likelihood_score_laplace_smoothing
    if args.smoothing == 'laplace_lindstone':
        scoring_function = get_query_likelihood_score_laplace_lindstone_smoothing
    if args.smoothing == 'jelinek_mercer':
        scoring_function = get_query_likelihood_score_jelinek_mercer_smoothing
    if args.smoothing == 'dirichlet':
        scoring_function = get_query_likelihood_score_dirichlet_smoothing

    # query likelihood scores for each claim-doc combination
    docs_with_query_likelihood_scores = [scoring_function(claim_terms, doc_with_terms) for
                                         doc_with_terms in
                                         doc_candidates.items()]

    # zero values lead to random retrievals if all documents evaluate to zero, so might rather want to show no results
    if (args.remove_zero_likelihood):
        docs_with_query_likelihood_scores = list(filter(lambda x: x[1] != 0, docs_with_query_likelihood_scores))

    # sort by query likelihood and limit to top results
    docs_with_query_likelihood_scores.sort(key=itemgetter(1), reverse=True)
    result_docs = docs_with_query_likelihood_scores[:DOCS_TO_RETRIEVE_PER_CLAIM]

    result_directory = '{}{}/'.format(RETRIEVED_PROBABILISTIC_DIRECTORY, args.smoothing or 'no_smoothing')
    display_or_store_result(claim, claim_id, result_docs, result_directory, args.print)


def retrieve_documents_for_claim_row(claim_row: tuple):
    claim_id = claim_row[1]['id']
    claim = claim_row[1]['claim']
    retrieve_documents_for_claim(claim, claim_id)


def retrieve_documents_for_all_claims():
    claims = get_all_claims(dataset=args.dataset)

    pool = get_process_pool()
    if (args.limit):
        pool.map(retrieve_documents_for_claim_row, claims.head(n=16).iterrows())
    else:
        pool.map(retrieve_documents_for_claim_row, claims.iterrows())


if __name__ == '__main__':
    start_time = time.time()
    if args.id:
        claim = get_claim_row(args.id, dataset=args.dataset)
        document = retrieve_documents_for_claim_row((None, claim))
    else:
        retrieve_documents_for_all_claims()
    print('Finished retrieval after {:.2f} seconds'.format(time.time() - start_time))

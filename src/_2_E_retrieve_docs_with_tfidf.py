import argparse
import re
import time
from multiprocessing import cpu_count, Pool
from operator import itemgetter

import pandas as pd
from termcolor import colored

from dataaccess.constants import GENERATED_IDF_PATH, DATA_TRAINING_PATH, RETRIEVED_TFIDF_DIRECTORY, \
    CLAIMS_COLUMNS_LABELED
from dataaccess.constants import get_inverted_index_shard_id, get_shard_path
from dataaccess.json_io import read_jsonl_and_map_to_df, read_dict_from_json, write_list_to_jsonl
from documentretrieval.term_processing import process_normalise_tokenise_filter
from documentretrieval.wiki_page_retrieval import retrieve_wiki_page

parser = argparse.ArgumentParser()
parser.add_argument('--id', help='ID of a claim to retrieve for test purposes (if defined, process only this one)', type=int)
parser.add_argument('--limit', help='only use subset for the first 10 claims', action='store_true', default=False)
parser.add_argument('--print', help='print results rather than storing on disk', action='store_true', default=False)
args = parser.parse_args()

DOCS_PER_CLAIM = 5

words_with_idf = read_jsonl_and_map_to_df(GENERATED_IDF_PATH, ['word', 'idf']).set_index('word', drop=False)
claims = read_jsonl_and_map_to_df(DATA_TRAINING_PATH, CLAIMS_COLUMNS_LABELED).set_index('id', drop=False)
cached_inverted_index_shards = {}


# Make slight changes to claim so that it has the same formatting as the text in wiki-pages, to ensure
# the tokenising and filtering work under under the same conditions
def preprocess_claim(claim: str) -> str:
    return re.sub(r'([.,!?;])', r' \1 ', claim)


def read_inverted_index_shard(shard_id: int) -> dict:
    shard_path = get_shard_path(shard_id)
    shard = read_dict_from_json(shard_path)
    return shard[1]


def get_occurrences(term: str) -> list:
    shard_id = get_inverted_index_shard_id(term)
    shard = cached_inverted_index_shards.setdefault(shard_id, read_inverted_index_shard(shard_id))
    return shard[term]


def compute_tfidf_similariy(doc_with_coordination_terms: list) -> tuple:
    page_id = doc_with_coordination_terms[0]
    coordination_terms_for_doc = doc_with_coordination_terms[1]
    similarity_score = 0
    for tf, idf in coordination_terms_for_doc:
        similarity_score += tf * idf

    return (page_id, similarity_score)


def retrieve_document_for_claim(claim_row: pd.Series):
    claim_id = claim_row['id']
    claim = claim_row['claim']

    print(colored('Retrieving documents for claim "{}"'.format(claim)))
    preprocessed_claim = preprocess_claim(claim)
    claim_terms = process_normalise_tokenise_filter(preprocessed_claim)

    doc_candidates = {}
    # For all terms, group by document and gather TF and IDF values
    for term in claim_terms:
        idf = words_with_idf.loc[term]['idf']
        occurrences = get_occurrences(term)
        for occurrence in occurrences:
            page_id = occurrence[0]
            tf = occurrence[1]
            doc_candidates.setdefault(page_id, []).append((tf, idf))

    # Compute TF-IDF similarity for docs
    docs_with_similarity_scores = list(map(compute_tfidf_similariy, doc_candidates.items()))

    # Sort by similarity and limit to top results
    docs_with_similarity_scores.sort(key=itemgetter(1), reverse=True)
    result_docs = docs_with_similarity_scores[:DOCS_PER_CLAIM]

    if (args.print):
        print(colored('Results for claim "{}":'.format(claim), attrs=['bold']))
        for doc in result_docs:
            page_id = doc[0]
            wiki_page = retrieve_wiki_page(page_id)
            print(wiki_page)
    else:
        result_path = '{}{}'.format(RETRIEVED_TFIDF_DIRECTORY, claim_id)
        write_list_to_jsonl(result_path, result_docs)


def retrieve_documents_for_all_claims():
    print(('Detected {} CPUs'.format(cpu_count())))
    pool = Pool(processes=cpu_count())

    # Process in multiple blocking processes4
    # FIXME iter
    if (args.limit):
        pool.map(retrieve_document_for_claim, claims.head(n=10).iterrows())
    else:
        pool.map(retrieve_document_for_claim, claims.iterrows())


if __name__ == '__main__':
    start_time = time.time()
    if (args.id):
        claim = claims.loc[args.id]
        document = retrieve_document_for_claim(claim)
    else:
        retrieve_documents_for_all_claims()
    print('Finished retrrieval after {:.2f} seconds'.format(time.time() - start_time))

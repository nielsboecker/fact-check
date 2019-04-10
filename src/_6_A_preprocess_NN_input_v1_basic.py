import argparse
from itertools import chain

import pandas as pd

from dataaccess.access_claims import get_all_claims
from dataaccess.access_wiki_page import retrieve_wiki_page
from dataaccess.files_constants import GENERATED_NN_PREPROCESSED_DATA
from dataaccess.files_io import write_pickle
from documentretrieval.claim_processing import preprocess_claim_text
from documentretrieval.data_constants import PREPROCESSED_DATA_COLUMNS_V1
from documentretrieval.term_processing import preprocess_doc_text
from relevance.embeddings import transform_sentence_to_vector, get_vector_difference
from relevance.evidence_relevance import get_evidence_page_line_map
from util.theads_processes import get_process_pool

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='only use subset of data', action='store_true')
parser.add_argument('--dataset', type=str, choices=['train', 'dev', 'test'], required=True)
parser.add_argument('--cores', type=int, default=None, help='Limit number of cores (optional)')
args = parser.parse_args()


def transform_NN_input(claim_text: str, line_text: str):
    # remove punctuation that are otherwise part of tokens
    preprocessed_claim = preprocess_claim_text(claim_text)
    # remove artifacts like -LRB- etc.
    preprocessed_line = preprocess_doc_text(line_text)

    claim_vector = transform_sentence_to_vector(preprocessed_claim, args.debug)
    line_vector = transform_sentence_to_vector(preprocessed_line, args.debug)
    combined_claim_line_vector = get_vector_difference(claim_vector, line_vector)

    return combined_claim_line_vector


def get_num_coordination_terms(x: list, y: list) -> int:
    coordination_terms = [term for term in x if x in y]
    return len(coordination_terms)


def preprocess_claim(claim_row: pd.Series) -> list:
    claim_id, verifiable, label, claim, evidence = claim_row[1].values
    if not verifiable == 'VERIFIABLE':
        return []
    print('Preprocessing docs for claim [{}]'.format(claim_id))

    # output will be the same for all evidence items belonging to this claim
    output = 1 if label == 'SUPPORTS' else 0
    preprocessed_pairs = []
    evidence_map = get_evidence_page_line_map(claim_id, args.dataset)

    for page_id, relevant_line_ids in evidence_map.items():
        wiki_page = retrieve_wiki_page(page_id)
        for line_id in relevant_line_ids:
            line_text = wiki_page.lines[line_id].text
            input = transform_NN_input(claim, line_text)
            preprocessed_pairs.append((claim_id, page_id, line_id, input, output))

    return preprocessed_pairs


if __name__ == '__main__':
    training_data = get_all_claims(args.dataset)
    if args.debug:
        training_data = training_data.head(n=5)

    pool = get_process_pool(args.cores)
    partial_results = pool.map(preprocess_claim, training_data.iterrows())
    print('Merging partial results...')
    preprocessed = list(chain.from_iterable(partial_results))

    preprocessed_df = pd.DataFrame.from_records(preprocessed, columns=PREPROCESSED_DATA_COLUMNS_V1)
    output_path = GENERATED_NN_PREPROCESSED_DATA.format(args.dataset, 'v1')
    write_pickle(output_path, preprocessed_df)

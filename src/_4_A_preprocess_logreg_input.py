import argparse
import os
from itertools import chain

import numpy as np
import pandas as pd

from dataaccess.access_claims import get_claim, claim_is_verifiable
from dataaccess.access_wiki_page import retrieve_wiki_page
from dataaccess.files_constants import GENERATED_LR_PREPROCESSED_TRAINING_DATA, GENERATED_LR_PREPROCESSED_DEV_DATA
from dataaccess.files_io import write_pickle
from documentretrieval.data_constants import PREPROCESSED_DATA_COLUMNS_V1
from model.wiki_document import WikiDocument
from relevance.embeddings import transform_LR_input
from relevance.evidence_relevance import get_evidence_page_line_map, is_relevant
from util.theads_processes import get_process_pool

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='don\'t load GloVe embeddings, use fake vectors', action='store_true')
parser.add_argument('--dataset', type=str, choices=['train', 'train_all', 'dev'], default='train')
parser.add_argument('--file', type=str, help='use this file (overrides dataset)')
parser.add_argument('--cores', type=int, default=None, help='Limit number of cores (optional)')
args = parser.parse_args()


def preprocess_doc(claim_id: int, claim: str, doc: WikiDocument, evidence_map: dict):
    preprocessed_lines = []
    for line in doc.lines:
        if not line.text:
            # Many docs in the FEVER dataset have empty lines
            continue
        input = transform_LR_input(claim, line.text, args.debug)
        output = 1 if is_relevant(doc.id, line.id, evidence_map) else 0
        preprocessed_lines.append((claim_id, doc.id, line.id, input, output))
    return preprocessed_lines


def preprocess_claim_with_doc(claim_with_docs: tuple) -> list:
    partial_result = []
    claim_id = claim_with_docs[0]
    # remove any NOT_VERIFIABLE claims that were processed earlier
    if not claim_is_verifiable(claim_id, dataset=args.dataset):
        return []

    claim = get_claim(claim_id, dataset=args.dataset)
    evidence_map = get_evidence_page_line_map(claim_id, args.dataset)

    retrieved_doc_ids = set(claim_with_docs[1])
    evidence_doc_ids = evidence_map.keys()
    docs_for_training = retrieved_doc_ids.union(evidence_doc_ids)
    print('Preprocessing docs for claim [{}]: {}'.format(claim_id, docs_for_training))

    docs = [retrieve_wiki_page(id) for id in docs_for_training]
    for doc in docs:
        partial_result.extend(preprocess_doc(claim_id, claim, doc, evidence_map))

    return partial_result


if __name__ == '__main__':
    in_path = None

    if args.file:
      in_path = args.file

    # overriden from args.file
    elif args.dataset == 'train':
        in_path = './submission/retrieved_train/Q3_laplace_lindstone_0.01.csv'
    elif args.dataset == 'train_all':
        in_path = './submission/retrieved_train/Q3_laplace_lindstone_0.01_10000_claims.csv'
    elif args.dataset == 'dev':
        in_path = './submission/retrieved_dev/Q3_laplace_lindstone_0.01.csv'
    elif args.dataset == 'dev_all':
        in_path = './submission/retrieved_dev/Q3_laplace_lindstone_0.01_10000_claims.csv.csv'

    claims_and_retrieved_docs = pd.read_csv(in_path, delimiter=',', quotechar='|', header=0, index_col=0)
    pool = get_process_pool(cores=args.cores)

    # 10,000 claims + sentences too much to keep in memory at once
    if args.dataset.endswith('all') or (args.file and args.file.endswith('all')):
        print('batch processing data')
        claims_split = np.array_split(claims_and_retrieved_docs, 20)
        for i, batch in enumerate(claims_split):
            partial_results = pool.map(preprocess_claim_with_doc, batch.iterrows())
            print('Merging partial results...')
            preprocessed = list(chain.from_iterable(partial_results))

            training_data = pd.DataFrame.from_records(preprocessed, columns=PREPROCESSED_DATA_COLUMNS_V1)
            output_path = GENERATED_LR_PREPROCESSED_TRAINING_DATA if args.dataset.startswith('train') \
                else GENERATED_LR_PREPROCESSED_DEV_DATA
            output_path += str(i)
            write_pickle(output_path, training_data)

    # just small subset of data
    else:
        partial_results = pool.map(preprocess_claim_with_doc, claims_and_retrieved_docs.iterrows())
        print('Merging partial results...')
        preprocessed = list(chain.from_iterable(partial_results))

        training_data = pd.DataFrame.from_records(preprocessed, columns=PREPROCESSED_DATA_COLUMNS_V1)
        output_path = GENERATED_LR_PREPROCESSED_TRAINING_DATA if args.dataset.startswith('train') \
            else GENERATED_LR_PREPROCESSED_DEV_DATA
        if args.file:
            output_path += os.path.basename(args.file)
        write_pickle(output_path, training_data)

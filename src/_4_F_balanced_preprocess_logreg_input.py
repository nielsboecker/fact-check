import argparse
from itertools import chain

import numpy as np
import pandas as pd

from dataaccess.access_claims import get_claim, claim_is_verifiable
from dataaccess.access_glove_embeddings import get_embedding
from dataaccess.access_wiki_page import retrieve_wiki_page
from dataaccess.files_constants import GENERATED_LR_PREPROCESSED_TRAINING_DATA, GENERATED_LR_PREPROCESSED_DEV_DATA
from dataaccess.files_io import write_pickle
from documentretrieval.claim_processing import preprocess_claim_text
from documentretrieval.data_constants import PREPROCESSED_DATA_COLUMNS
from documentretrieval.term_processing import preprocess_doc_text
from relevance.evidence_relevance import get_evidence_page_line_map, get_irrelevant_line
from util.theads_processes import get_process_pool
from util.vector_algebra import get_min_max_vectors

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='don\'t load GloVe embeddings, use fake vectors', action='store_true')
parser.add_argument('--dataset', type=str, choices=['train', 'train_all', 'dev', 'dev_all'], default='train')
parser.add_argument('--file', type=str, help='use this file (overrides dataset)')
args = parser.parse_args()


def transform_sentence_to_vector(sentence: str):
    # Refer to https://arxiv.org/pdf/1607.00570.pdf
    embeddings = [get_embedding(term, args.debug) for term in sentence.split()]
    min_values, max_values = get_min_max_vectors(embeddings)
    sentence_vector = np.concatenate((min_values, max_values), axis=0)
    assert sentence_vector.shape == (600,)
    return sentence_vector


def create_feature_vector(claim_vector: np.array, line_vector: np.array) -> np.array:
    return claim_vector - line_vector


def transform_input(claim_text: str, line_text: str):
    # remove punctuation that are otherwise part of tokens
    preprocessed_claim = preprocess_claim_text(claim_text)
    # remove artifacts like -LRB- etc.
    preprocessed_line = preprocess_doc_text(line_text)

    claim_vector = transform_sentence_to_vector(preprocessed_claim)
    line_vector = transform_sentence_to_vector(preprocessed_line)

    feature_vector = create_feature_vector(claim_vector, line_vector)
    return feature_vector


def preprocess_claim_with_doc(claim_with_docs: tuple) -> list:
    claim_id = claim_with_docs[0]
    # remove any NOT_VERIFIABLE claims that were processed earlier
    if not claim_is_verifiable(claim_id, dataset=args.dataset):
        return []

    claim = get_claim(claim_id, dataset=args.dataset)
    evidence_map = get_evidence_page_line_map(claim_id, args.dataset)
    print('Preprocessing docs for claim [{}]: {}'.format(claim_id, evidence_map.keys()))

    preprocessed_pairs = []
    for page_id, relevant_line_ids in evidence_map.items():
        wiki_page = retrieve_wiki_page(page_id)
        for line_id in relevant_line_ids:
            # add the relevant claim/sentence pair...
            positive_line = wiki_page.lines[line_id]
            positive_input = transform_input(claim, positive_line.text)
            preprocessed_pairs.append((claim_id, page_id, line_id, positive_input, 1))

            # ...and, to keep it balanced, one irrelevant sample
            negative_line = get_irrelevant_line(wiki_page, relevant_line_ids)
            negative_input = transform_input(claim, negative_line.text)
            preprocessed_pairs.append((claim_id, page_id, negative_line.id, negative_input, 0))

    return preprocessed_pairs


if __name__ == '__main__':
    in_path = None

#    if args.file:
#      in_path = args.file

#    # overriden from args.file
    if args.dataset == 'train':
        in_path = './submission/retrieved_train/Q3_laplace_lindstone_0.01.csv'
    elif args.dataset == 'train_all':
        in_path = './submission/retrieved_train/Q3_laplace_lindstone_0.01_10000_claims.csv'
    elif args.dataset == 'dev':
        in_path = './submission/retrieved_dev/Q3_laplace_lindstone_0.01.csv'
    elif args.dataset == 'dev_all':
        in_path = './submission/retrieved_dev/Q3_laplace_lindstone_0.01_10000_claims.csv'

    claims_and_retrieved_docs = pd.read_csv(in_path, delimiter=',', quotechar='|', header=0, index_col=0)
    if args.debug:
        claims_and_retrieved_docs = claims_and_retrieved_docs.head(n=2)

    pool = get_process_pool(cores=14)

#    # 10,000 claims + sentences too much to keep in memory at once
#    if args.dataset.endswith('all') or (args.file and args.file.endswith('all')):
#        print('batch processing data')
#        claims_split = np.array_split(claims_and_retrieved_docs, 20)
#        for i, batch in enumerate(claims_split):
#            partial_results = pool.map(preprocess_claim_with_doc, batch.iterrows())
#            print('Merging partial results...')
#            preprocessed = list(chain.from_iterable(partial_results))
#
#            training_data = pd.DataFrame.from_records(preprocessed, columns=PREPROCESSED_DATA_COLUMNS)
#            output_path = GENERATED_PREPROCESSED_TRAINING_DATA if args.dataset.startswith('train') \
#                else GENERATED_PREPROCESSED_DEV_DATA
#            output_path += str(i)
#            write_pickle(output_path, training_data)
#
#    # just small subset of data
#    else:
    partial_results = pool.map(preprocess_claim_with_doc, claims_and_retrieved_docs.iterrows())
    print('Merging partial results...')
    preprocessed = list(chain.from_iterable(partial_results))

    training_data = pd.DataFrame.from_records(preprocessed, PREPROCESSED_DATA_COLUMNS)
    output_path = GENERATED_LR_PREPROCESSED_TRAINING_DATA if args.dataset.startswith('train') \
        else GENERATED_LR_PREPROCESSED_DEV_DATA
#    if args.file:
#        output_path += os.path.basename(args.file)
    write_pickle(output_path, training_data)

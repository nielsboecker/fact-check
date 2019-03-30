# Note: no need for multiprocessing here, as only loading the embeddings
# into memory takes time. After that, only a few quick look-ups remain.
import argparse

import numpy as np
import pandas as pd

from dataaccess.access_glove_embeddings import get_embedding
from dataaccess.access_training_data import get_training_claim, get_training_claim_row, is_verifiable
from dataaccess.access_wiki_page import retrieve_wiki_page
from dataaccess.constants import GENERATED_PREPROCESSED_TRAINING_DATA
from dataaccess.files_io import write_pickle
from documentretrieval.claim_processing import preprocess_text
from model.wiki_document import WikiDocument
from util.vector_algebra import get_min_max_vectors

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='don\'t load GloVe embeddings, use fake vectors', action='store_true')
parser.add_argument('--dataset', type=str, choices=['train', 'train_all', 'dev'], default='train')
#parser.add_argument('--in_file', type=str, required=True)
args = parser.parse_args()

PREPROCESSED_DATA_COLUMNS = ['claim_id', 'page_id', 'line_id', 'input_vector', 'expected_output']

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
    claim_vector = transform_sentence_to_vector(claim_text)
    line_vector = transform_sentence_to_vector(line_text)
    feature_vector = create_feature_vector(claim_vector, line_vector)
    return feature_vector


def is_relevant(doc_id: str, line_id: int, evidence_map: dict) -> bool:
    if doc_id in evidence_map:
        return line_id in evidence_map[doc_id]
    else:
        return False


def preprocess_doc(claim_id: int, claim: str, doc: WikiDocument, evidence_map: dict):
    preprocessed_lines = []
    for line in doc.lines:
        if not line.text:
            # Many docs in the FEVER dataset have empty lines
            continue
        input = transform_input(claim, line.text)
        output = 1 if is_relevant(doc.id, line.id, evidence_map) else 0
        preprocessed_lines.append((claim_id, doc.id, line.id, input, output))
    return preprocessed_lines


def get_evidence_page_line_map(claim_id: int) -> dict:
    mapping = {}
    evidence = get_training_claim_row(claim_id)['evidence'][0]
    for _, _, page_id, line_id in evidence:
        mapping.setdefault(page_id, []).append(line_id)
    return mapping


if __name__ == '__main__':
    in_path = None
    if args.dataset == 'train':
        in_path = './submission/retrieved_train/Q3_laplace_lindstone_0.01.csv'
    elif args.dataset == 'train_all':
        in_path = './submission/retrieved_train/Q3_laplace_lindstone_0.01_10000_claims.csv'
    elif args.dataset == 'dev':
        in_path = './submission/retrieved_dev/Q3_laplace_lindstone_0.01.csv'

    claims_and_retrieved_docs = pd.read_csv(in_path, delimiter=',', quotechar='|', header=0, index_col=0)
    preprocessed = []

    for claim_with_docs in claims_and_retrieved_docs.iterrows():
        claim_id = claim_with_docs[0]
        # remove any NOT_VERIFIABLE claims that were processed earlier
        if not is_verifiable(claim_id):
            continue

        claim = preprocess_text(get_training_claim(claim_id))
        evidence_map = get_evidence_page_line_map(claim_id)

        retrieved_doc_ids = set(claim_with_docs[1])
        evidence_doc_ids = evidence_map.keys()
        docs_for_training = retrieved_doc_ids.union(evidence_doc_ids)
        print('Preprocessing docs for claim [{}]: {}'.format(claim_id, docs_for_training))

        docs = [retrieve_wiki_page(id) for id in docs_for_training]
        for doc in docs:
            preprocessed.extend(preprocess_doc(claim_id, claim, doc, evidence_map))

    training_data = pd.DataFrame.from_records(preprocessed, columns=PREPROCESSED_DATA_COLUMNS)
    write_pickle(GENERATED_PREPROCESSED_TRAINING_DATA, training_data)

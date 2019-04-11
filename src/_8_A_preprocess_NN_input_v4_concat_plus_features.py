import argparse
from itertools import chain

import numpy as np
import pandas as pd

from dataaccess.access_claims import get_all_claims
from dataaccess.access_wiki_page import retrieve_wiki_page
from dataaccess.files_constants import GENERATED_NN_PREPROCESSED_DATA
from dataaccess.files_io import write_pickle
from documentretrieval.claim_processing import preprocess_claim_text
from documentretrieval.data_constants import PREPROCESSED_DATA_COLUMNS_V2
from documentretrieval.document_processing import preprocess_doc_title
from documentretrieval.term_processing import preprocess_doc_text
from relevance.embeddings import transform_sentence_to_vector, get_vector_difference
from relevance.evidence_relevance import get_evidence_page_line_map
from util.theads_processes import get_process_pool

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='only use subset of data', action='store_true')
parser.add_argument('--cores', type=int, default=None, help='Limit number of cores (optional)')
parser.add_argument('--dataset', type=str, choices=['train', 'dev', 'test'], required=True)
args = parser.parse_args()


def transform_NN_input(claim_text: str, combined_evidence: str, num_evidence_docs_for_claim: int, 
                       num_references: int, num_evidence_items: int, num_coordination_terms_evidence_claim: int,
                       num_coordination_terms_titles_claim: int, avg_sentence_position: float, num_evidence_words: int):
    # remove punctuation that are otherwise part of tokens
    preprocessed_claim = preprocess_claim_text(claim_text)
    # remove artifacts like -LRB- etc.
    preprocessed_line = preprocess_doc_text(combined_evidence)

    claim_vector = transform_sentence_to_vector(preprocessed_claim, args.debug)
    line_vector = transform_sentence_to_vector(preprocessed_line, args.debug)
    combined_claim_line_vector = get_vector_difference(claim_vector, line_vector)
    
    additional_features = np.array((num_evidence_docs_for_claim, num_references, num_evidence_items,
                                    num_coordination_terms_evidence_claim, num_coordination_terms_titles_claim,
                                    avg_sentence_position, num_evidence_words))

    return np.concatenate((combined_claim_line_vector, additional_features))


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

    evidence_sentences = []
    num_evidence_docs_for_claim = len(evidence_map.keys())
    num_references = 0
    num_evidence_items = 0
    num_evidence_words = 0
    num_coordination_terms_evidence_claim = 0
    num_coordination_terms_titles_claim = 0
    evidence_sentence_positions = []

    # concat evidence (can be from multiple wiki_pages and/or lines)
    for page_id, relevant_line_ids in evidence_map.items():
        wiki_page = retrieve_wiki_page(page_id)
        evidence_sentences.extend([wiki_page.lines[id].text for id in relevant_line_ids])

        # count metrics and subtract features
        for line_id in relevant_line_ids:
            line = wiki_page.lines[line_id]
            line_text = line.text
            num_evidence_words += len(line_text.split())
            num_references += len(line.anchors)
            num_evidence_items += 1
            evidence_sentence_positions.append(line.id)
            num_coordination_terms_evidence_claim += get_num_coordination_terms(line_text, preprocess_claim_text(claim).split())
            num_coordination_terms_titles_claim += get_num_coordination_terms(line_text, preprocess_doc_title(page_id))


    combined_evidence = ' '.join(evidence_sentences)
    avg_sentence_position = np.mean(evidence_sentence_positions)

    input = transform_NN_input(claim, combined_evidence, num_evidence_docs_for_claim, num_references,
                               num_evidence_items, num_coordination_terms_evidence_claim,
                               num_coordination_terms_titles_claim, avg_sentence_position, num_evidence_words)
    preprocessed_pairs.append((claim_id, input, output))

    return preprocessed_pairs


if __name__ == '__main__':
    training_data = get_all_claims(args.dataset)
    if args.debug:
        training_data = training_data.head(n=3)

    pool = get_process_pool(args.cores)
    partial_results = pool.map(preprocess_claim, training_data.iterrows())
    print('Merging partial results...')
    preprocessed = list(chain.from_iterable(partial_results))

    preprocessed_df = pd.DataFrame.from_records(preprocessed, columns=PREPROCESSED_DATA_COLUMNS_V2)
    output_path = GENERATED_NN_PREPROCESSED_DATA.format(args.dataset, 'v4')
    write_pickle(output_path, preprocessed_df)

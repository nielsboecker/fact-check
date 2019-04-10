import argparse
from itertools import chain

import pandas as pd

from dataaccess.access_claims import get_all_claims
from dataaccess.access_wiki_page import retrieve_wiki_page
from dataaccess.files_constants import GENERATED_ENTAILMENT_PREPROCESSED_DATA
from dataaccess.files_io import write_dataframe_to_csv
from documentretrieval.term_processing import recreate_punctuation_in_doc_text
from relevance.evidence_relevance import get_evidence_page_line_map
from util.theads_processes import get_process_pool

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='only use subset of data', action='store_true')
parser.add_argument('--cores', type=int, default=2, help='cores to run on')
parser.add_argument('--dataset', type=str, choices=['train', 'dev'], default='train')
args = parser.parse_args()


def preprocess_claim(claim_row: pd.Series) -> list:
    claim_id, verifiable, label, claim, evidence = claim_row[1].values
    if not verifiable == 'VERIFIABLE':
        return []
    print('Preprocessing docs for claim [{}]'.format(claim_id))

    # output will be the same for all evidence items belonging to this claim
    #label = 'entails' if label == 'SUPPORTS' else 'neutral'
    label = 1 if label == 'SUPPORTS' else 0
    preprocessed_pairs = []
    evidence_map = get_evidence_page_line_map(claim_id, args.dataset)
    evidence_sentences = []

    for page_id, relevant_line_ids in evidence_map.items():
        wiki_page = retrieve_wiki_page(page_id)
        evidence_sentences.extend([wiki_page.lines[id].text for id in relevant_line_ids])

    premise = ' '.join(evidence_sentences)
    premise = recreate_punctuation_in_doc_text(premise)
    hypothesis = recreate_punctuation_in_doc_text(claim)
    pair = {'label': label, 'sentence1': premise, 'sentence2': hypothesis, 'claim_id': claim_id}
    preprocessed_pairs.append(pair)

    return preprocessed_pairs


if __name__ == '__main__':
    training_data = get_all_claims(args.dataset)
    if args.debug:
        training_data = training_data.head(n=5)

    pool = get_process_pool(args.cores)
    partial_results = pool.map(preprocess_claim, training_data.iterrows())
    print('Merging partial results...')
    preprocessed = list(chain.from_iterable(partial_results))

    output_path = GENERATED_ENTAILMENT_PREPROCESSED_DATA.format(args.dataset)
    preprocessed_df = pd.DataFrame.from_records(preprocessed, columns=['label', 'sentence1', 'sentence2', 'claim_id'])
    write_dataframe_to_csv(output_path, preprocessed_df, sep='\t', header=False)

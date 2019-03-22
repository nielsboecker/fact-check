import math
import time
from collections import Counter

import argparse

from dataaccess.constants import GENERATED_DOCUMENT_NORMS_MAPPING, get_wiki_batch_path
from dataaccess.json_io import read_jsonl_and_map_to_df, write_dict_to_json
from documentretrieval.access_inverted_index import get_index_entry_for_term
from documentretrieval.document_processing import filter_documents, reduce_document_to_text_column
from documentretrieval.term_processing import process_normalise_tokenise_filter
from util.theads_processes import get_process_pool

parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="only use subset of data", action="store_true")
parser.add_argument('--variant', help='TF weighting variant', choices=['raw_count', 'relative'], default='relative')
args = parser.parse_args()


def get_tfidf_vector_norm(text: list):
    # Note: this takes the TF directly from the text argument
    word_count = Counter(text)
    accumulated_tfidf_values_for_words = []
    for word, count in word_count.items():
        tf = count if args.variant == 'raw_count' else float(count) / len(text)
        index_entry = get_index_entry_for_term(word)
        idf = index_entry['idf']
        tfidf_value = tf * idf
        accumulated_tfidf_values_for_words.append(tfidf_value)
    norm = math.sqrt(sum([i ** 2 for i in accumulated_tfidf_values_for_words]))

    if(args.debug):
        print('Computed norm {} for doc \n{}'.format(norm, text))
    return norm


def generate_document_norm_mapping_for_batch(batch_id: int) -> dict:
    batch_file_path = get_wiki_batch_path(batch_id)
    all_articles = read_jsonl_and_map_to_df(batch_file_path, ['id', 'text'])
    filtered_articles = filter_documents(all_articles)
    # filtered_articles.set_index('id', drop=False)
    if (args.debug):
        filtered_articles = filtered_articles.head(n=10)

    partial_document_norm_mappings = {}
    for raw_doc_row in filtered_articles.iterrows():
        page_id = raw_doc_row[1]['id']
        filtered_tokens = process_normalise_tokenise_filter(raw_doc_row[1]['text'])
        doc_norm = get_tfidf_vector_norm(filtered_tokens)
        partial_document_norm_mappings[page_id] = doc_norm

    return partial_document_norm_mappings


def generate_document_norm_mapping_all() -> dict:
    start_index_inclusive = 108 if args.debug else 1
    stop_index_exclusive = 110

    pool = get_process_pool()
    batch_partial_counts = pool.map(generate_document_norm_mapping_for_batch, range(start_index_inclusive, stop_index_exclusive))
    accumulated_document_word_mapping = {}
    for batch_result in batch_partial_counts:
        accumulated_document_word_mapping.update(batch_result)
    return accumulated_document_word_mapping


if __name__ == '__main__':
    start_time = time.time()
    document_norm_mapping = generate_document_norm_mapping_all()
    print('Finished generating document -> norm mapping after {:.2f} seconds'.format(time.time() - start_time))
    write_dict_to_json(GENERATED_DOCUMENT_NORMS_MAPPING, document_norm_mapping)

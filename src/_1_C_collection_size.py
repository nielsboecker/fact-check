import re
import time
from collections import Counter

import pandas as pd
from termcolor import colored

from json_io import read_jsonl_and_map_to_df, write_list_to_jsonl
from _1_A_word_frequency_count import get_wiki_batch_path, filter_articles


def get_word_counts(words: list) -> Counter:
    counter = Counter(words)
    print("Counted word frequencies for {:,} words ({:,} unique)".format(len(words), len(counter.keys())))
    return counter


def process_filter_count_batch(batch_id: int) -> int:
    batch_file_path = get_wiki_batch_path(batch_id)
    all_articles = read_jsonl_and_map_to_df(batch_file_path, ['text'])
    filtered_articles = filter_articles(all_articles)
    print('Using {} articles after filtering'.format(len(filtered_articles)))
    return len(filtered_articles)


def process_count_all() -> int:
    start_index_inclusive = 1
    stop_index_exclusive = 110
    accumulated_collection_size = 0
    for id in range(start_index_inclusive, stop_index_exclusive):
        accumulated_collection_size += process_filter_count_batch(id)
    return accumulated_collection_size


if __name__ == '__main__':
    start_time = time.time()
    collection_size = process_count_all()
    print(colored('After filtering, collection size is {}.'.format(collection_size)))
    print('Finished processing after {:.2f} seconds'.format(time.time() - start_time))

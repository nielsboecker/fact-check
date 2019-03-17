import argparse
import logging
import os
import re
import sys
import time
from collections import Counter
import concurrent.futures
import multiprocessing
from multiprocessing import Pool, cpu_count

import pandas as pd
from termcolor import colored

from constants import DATA_WIKI_PATH, GENERATED_COUNTS_PATH, GENERATED_IDFS_PATH
from jsonl_io import read_jsonl_and_map_to_df, write_list_to_jsonl


#### parser = argparse.ArgumentParser()
#### #parser.add_argument("--debug", help="only use subset of data", action="store_true")
#### parser.add_argument("-p", "--processes", type=int, default=4, help="number of processes to spawn")
#### args = parser.parse_args()

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
# console = logging.StreamHandler()
# console.setFormatter(fmt)
# logger.addHandler(console)

TERM_COLOURS = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']


def filter_articles(articles: pd.DataFrame) -> pd.DataFrame:
    min_article_length = 20
    is_long_enough = articles['text'].str.len() > min_article_length
    return articles[is_long_enough]


def parse_article_text(articles: pd.DataFrame) -> pd.DataFrame:
    return articles['text']


def preprocess_article(article: str) -> str:
    return article.replace('-LRB-', '').replace('-RRB-', '')


def tokenise_article(article: str) -> list:
    return re.split(r'\s+|-', article)


def filter_tokens(tokens: list) -> list:
    regex = re.compile(r'^[a-zA-Z0-9]+$')
    filtered_tokens = filter(regex.search, tokens)
    return list(filtered_tokens)


def normalise_article(article):
    return article.lower()


def process_normalise_tokenise_filter(raw_article: str) -> list:
    article = preprocess_article(raw_article)
    normalised_article = normalise_article(article)
    all_tokens = tokenise_article(normalised_article)
    return filter_tokens(all_tokens)


def process_generate_idfs_batch(id: int) -> Counter:
    colour = TERM_COLOURS[id % len(TERM_COLOURS)]
    print(colored('Start processing batch #{}'.format(id), colour, attrs=['bold']))

    start_time = time.time()

    batch_file_path = '{}wiki-{:03}.jsonl'.format(DATA_WIKI_PATH, id)
    all_articles = read_jsonl_and_map_to_df(batch_file_path, ['text'])
    filtered_articles = filter_articles(all_articles)
    # print('Using {} articles after filtering'.format(len(filtered_articles)))
    article_texts = parse_article_text(filtered_articles)

    accumulated_batch_idfs = Counter()

    for index, raw_article in enumerate(article_texts):
        filtered_tokens = process_normalise_tokenise_filter(raw_article)
        # use set to prevent multiple occurrences of word in doc
        words_set = set(filtered_tokens)

        if (index % 5000 == 0):
            print(colored('Processing document [{} / {}] of batch #{}...'.format(index, len(article_texts), id), colour))

        # count for included words will be one
        words_in_doc = Counter(words_set)
        accumulated_batch_idfs += words_in_doc

    print(colored('Finished processing batch #{} after {:.2f} seconds'.format(id, time.time() - start_time), colour, attrs=['bold']))
    return accumulated_batch_idfs


def generate_idfs_all() -> list:
    start_index_inclusive = 1
    stop_index_exclusive = 110

    num_processes = cpu_count() # max(cpu_count() - 2, 2)
    print(colored('Detected {} CPUs, spawing {} processes'.format(cpu_count(), num_processes), attrs=['bold']))
    pool = Pool(processes=num_processes)

    # blocks until the result is ready
    batch_idfs_results = pool.map(process_generate_idfs_batch, range(start_index_inclusive, stop_index_exclusive))
    pool.close()

    print('Merging {} partial results...'.format(len(batch_idfs_results)))
    accumulated_all_idfs = Counter()
    for batch_result in batch_idfs_results:
        accumulated_all_idfs += batch_result

    return accumulated_all_idfs.most_common()


def export_result(result: list):
    write_list_to_jsonl(GENERATED_IDFS_PATH, result)


if __name__ == '__main__':
    start_time = time.time()

    words_and_idfs = generate_idfs_all()
    print(colored('Counted IDFs of {:,} words'.format(len(words_and_idfs)), attrs=['bold']))
    print('Top 10 extract: {}'.format(words_and_idfs[0:10]))
    print('Finished processing after {:.2f} seconds'.format(time.time() - start_time))
    export_result(words_and_idfs)

    # Vocabulary size should be equal from the frequency count in task #1
    vocabulary = read_jsonl_and_map_to_df(GENERATED_COUNTS_PATH)[0]
    assert(len(vocabulary) == len(words_and_idfs))

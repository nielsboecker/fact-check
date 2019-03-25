import math
import time
from collections import Counter
from multiprocessing import Pool, cpu_count

import argparse
from documentretrieval.document_processing import filter_documents, reduce_document_to_text_column
from documentretrieval.term_processing import process_normalise_tokenise_filter
from dataaccess.constants import DATA_WIKI_PATH, GENERATED_COUNTS_PATH, GENERATED_IDF_PATH, TERM_COLOURS, \
    COLLECTION_FILTERED_DOCUMENTS_NUMBER
from dataaccess.json_io import read_jsonl_and_map_to_df, write_list_to_jsonl
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="only use subset of data", action="store_true")
args = parser.parse_args()


def process_generate_df_batch(id: int) -> Counter:
    colour = TERM_COLOURS[id % len(TERM_COLOURS)]
    print(colored('Start processing batch #{}'.format(id), colour, attrs=['bold']))

    start_time = time.time()

    batch_file_path = '{}wiki-{:03}.jsonl'.format(DATA_WIKI_PATH, id)
    all_articles = read_jsonl_and_map_to_df(batch_file_path, ['text'])
    filtered_articles = filter_documents(all_articles)
    article_texts = reduce_document_to_text_column(filtered_articles)

    accumulated_batch_idfs = Counter()

    for index, raw_article in enumerate(article_texts):
        filtered_tokens = process_normalise_tokenise_filter(raw_article)
        # use set to prevent multiple occurrences of word in doc
        words_set = set(filtered_tokens)

        if (index % 5000 == 0):
            print(
                colored('Processing document [{} / {}] of batch #{}...'.format(index, len(article_texts), id), colour))

        # count for included words will be one
        words_in_doc = Counter(words_set)
        accumulated_batch_idfs += words_in_doc

    print(colored('Finished processing batch #{} after {:.2f} seconds'.format(id, time.time() - start_time), colour,
                  attrs=['bold']))
    return accumulated_batch_idfs


def generate_df_all() -> list:
    start_index_inclusive = 1
    stop_index_exclusive = 3 if args.debug else 110
    # NOTE: If debug, IDF values will be wrong (because collection size isn't valid)

    num_processes = cpu_count()  # max(cpu_count() - 2, 2)
    print(colored('Detected {} CPUs, spawning {} processes'.format(cpu_count(), num_processes), attrs=['bold']))
    pool = Pool(processes=num_processes)

    # blocks until the result is ready
    batch_idfs_results = pool.map(process_generate_df_batch, range(start_index_inclusive, stop_index_exclusive))
    pool.close()

    print('Merging {} partial results...'.format(len(batch_idfs_results)))
    accumulated_all_idfs = Counter()
    for batch_result in batch_idfs_results:
        accumulated_all_idfs += batch_result

    return accumulated_all_idfs.most_common()


def get_words_with_idf(words_with_df: list) -> list:
    result = []
    for word_count in words_with_df:
        word = word_count[0]
        df = word_count[1]
        idf = math.log10(COLLECTION_FILTERED_DOCUMENTS_NUMBER / df)
        result.append((word, idf))
    return result


def export_result(result: list):
    write_list_to_jsonl(GENERATED_IDF_PATH, result)


if __name__ == '__main__':
    start_time = time.time()

    words_with_df = generate_df_all()
    print(colored('Counted frequencies of {:,} words'.format(len(words_with_df)), attrs=['bold']))

    words_with_idf = get_words_with_idf(words_with_df)
    print('Added inverse document frequencies')

    print('Top 10 extract: {}'.format(words_with_idf[0:10]))
    print('Finished processing after {:.2f} seconds'.format(time.time() - start_time))
    export_result(words_with_idf)

    # Vocabulary size should be equal from the frequency count in task #1
    vocabulary = read_jsonl_and_map_to_df(GENERATED_COUNTS_PATH)[0]
    assert (len(vocabulary) == len(words_with_idf))

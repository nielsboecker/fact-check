import time
from collections import Counter
from multiprocessing import cpu_count, Pool

from termcolor import colored

from dataaccess.constants import GENERATED_COUNTS_PATH, get_wiki_batch_path
from dataaccess.json_io import read_jsonl_and_map_to_df, write_list_to_jsonl
from documentretrieval.document_processing import filter_articles, parse_article_text
from documentretrieval.term_processing import process_normalise_tokenise_filter


def get_word_counts(words: list) -> Counter:
    counter = Counter(words)
    print("Counted word frequencies for {:,} words ({:,} unique)".format(len(words), len(counter.keys())))
    return counter


def process_count_batch(batch_id: int) -> Counter:
    batch_file_path = get_wiki_batch_path(batch_id)
    all_articles = read_jsonl_and_map_to_df(batch_file_path, ['text'])
    filtered_articles = filter_articles(all_articles)
    print('Using {} articles after filtering'.format(len(filtered_articles)))
    article_texts = parse_article_text(filtered_articles)

    combined_tokens = []
    for raw_article in article_texts:
        filtered_tokens = process_normalise_tokenise_filter(raw_article)
        combined_tokens.extend(filtered_tokens)
    return get_word_counts(combined_tokens)


def process_count_all() -> list:
    start_index_inclusive = 1
    stop_index_exclusive = 110
    accumulated_word_count = Counter()

    # Process in multiple blocking processes
    print(('Detected {} CPUs'.format(cpu_count())))
    pool = Pool(processes=cpu_count())
    batch_partial_counts = pool.map(process_count_batch, range(start_index_inclusive, stop_index_exclusive))
    for batch_result in batch_partial_counts:
        accumulated_word_count += batch_result
    return accumulated_word_count.most_common()


if __name__ == '__main__':
    start_time = time.time()
    word_count = process_count_all()
    print(colored('Counted frequencies of {:,} unique words'.format(len(word_count)), attrs=['bold']))
    print('Top 10 extract: {}'.format(word_count[0:10]))
    print('Finished processing after {:.2f} seconds'.format(time.time() - start_time))
    write_list_to_jsonl(GENERATED_COUNTS_PATH, word_count)

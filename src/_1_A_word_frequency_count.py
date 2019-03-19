import re
import time
from collections import Counter
from multiprocessing import cpu_count, Pool

import nltk
import pandas as pd
from constants import DATA_WIKI_PATH, GENERATED_COUNTS_PATH
from json_io import read_jsonl_and_map_to_df, write_list_to_jsonl
from nltk.corpus import stopwords
from termcolor import colored

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


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


# Filter words that are too short, not consisting of alphanumeric characters, or are stopwords
def filter_tokens(tokens: list) -> list:
    regex = re.compile(r'^[a-zA-Z0-9]{2,}$')
    filtered_tokens = filter(regex.search, tokens)
    filtered_tokens = [word for word in filtered_tokens if word not in stop_words]
    return list(filtered_tokens)


def normalise_article(article):
    return article.lower()


def get_word_counts(words: list) -> Counter:
    counter = Counter(words)
    print("Counted word frequencies for {:,} words ({:,} unique)".format(len(words), len(counter.keys())))
    return counter


def process_normalise_tokenise_filter(raw_article: str) -> list:
    article = preprocess_article(raw_article)
    normalised_article = normalise_article(article)
    all_tokens = tokenise_article(normalised_article)
    return filter_tokens(all_tokens)


def get_wiki_batch_path(batch_id):
    return '{}wiki-{:03}.jsonl'.format(DATA_WIKI_PATH, batch_id)


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


def export_result(result: list):
    write_list_to_jsonl(GENERATED_COUNTS_PATH, result)


if __name__ == '__main__':
    start_time = time.time()
    word_count = process_count_all()
    print(colored('Counted frequencies of {:,} unique words'.format(len(word_count)), attrs=['bold']))
    print('Top 10 extract: {}'.format(word_count[0:10]))
    print('Finished processing after {:.2f} seconds'.format(time.time() - start_time))
    export_result(word_count)

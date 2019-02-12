import jsonlines
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_EXAMPLE_JSONL = './data/example.jsonl'

DATA_COLUMNS = ['id', 'verifiable', 'label', 'claim', 'evidence']


def read_jsonl_and_map_to_df(jsonl_path: str, columns: list):
    items = []
    print(">>>>> start reading jsonl")
    with jsonlines.open(jsonl_path) as reader:
        for fact in reader:
            items.append(fact)
    itemsDF = pd.DataFrame(items, columns=columns)
    print(">>>>> finish reading jsonl")
    return itemsDF


def get_words_breakdown(df: pd.DataFrame):
    words = []
    for row in df:
        words.extend(row.split(" "))
    return words


def get_word_counts(words: list, lower: bool = True, sort: bool = True) -> list:
    if lower:
        words = list(map(lambda x: str(x).lower(), words))
    counter = Counter(words)
    if sort:
        return counter.most_common()
    return list(counter.items())


if __name__ == '__main__':
    df = read_jsonl_and_map_to_df(DATA_EXAMPLE_JSONL, DATA_COLUMNS)
    word_list = get_words_breakdown(df['claim'])
    word_counts = get_word_counts(word_list)
    print(word_counts)

    distinct_words = [count[0] for count in word_counts]
    distinct_counts = [count[1] for count in word_counts]

    indexes = np.arange(len(distinct_words))
    width = 1
    plt.bar(indexes, distinct_counts, width)
    plt.xticks(indexes + width * 0.5, distinct_words)
    plt.show()

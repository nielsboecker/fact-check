import jsonlines
from collections import Counter
import pandas as pd


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


def get_words_breakdown(df: pd.DataFrame, lower: bool = True):
    words = []
    for row in df:
        words.extend(row.split(" "))
    if lower:
        words = list(map(lambda x: str(x).lower(), words))
    return words


def get_word_count(words: list, sorted: bool = True):
    counter = Counter(words)
    if sorted:
        return counter.most_common()
    return list(counter.items())


if __name__ == '__main__':
    df = read_jsonl_and_map_to_df(DATA_EXAMPLE_JSONL, DATA_COLUMNS)
    words = get_words_breakdown(df['claim'])
    counts = get_word_count(words)
    print(counts)

from collections import Counter
import pandas as pd


def get_words_breakdown(df: pd.DataFrame):
    words = []
    for row in df:
        words.extend(row.split(" "))

    print("Extracted {:,} words from {:,} sentences.".format(len(words), len(df)))
    return words


def get_word_counts(words: list, lower: bool = True, sort: bool = True) -> list:
    if lower:
        words = list(map(lambda x: str(x).lower(), words))
    counter = Counter(words)

    print("Counted word frequencies for {:,} words ({:,} unique).".format(len(words), len(counter.keys())))
    if sort:
        return counter.most_common()
    return list(counter.items())

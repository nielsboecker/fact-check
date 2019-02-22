import re
from collections import Counter, Iterator
import pandas as pd
from src.constants import DATA_WIKI_PATH
from src.jsonl_io import read_jsonl_and_map_to_df

start_index_inclusive = 1
stop_index_exclusive = 2 # 110

def filter_articles(articles: pd.DataFrame) -> pd.DataFrame:
    is_not_empty = articles['text'] != ''
    return articles[is_not_empty]

def parse_article_text(articles: pd.DataFrame) -> pd.DataFrame:
    return articles['text']

def tokenise_article(article: str) -> list:
    return article.split()

def filter_tokens(tokens: list) -> list:
    regex = re.compile(r'^\d+$')
    filteredTokens = filter(regex.search, tokens)
    return list(filteredTokens)



for i in range(start_index_inclusive, stop_index_exclusive):
    wiki_path = '{}wiki-{:03}.jsonl'.format(DATA_WIKI_PATH, i)
    print('Reading "{}"'.format(wiki_path))

    all_articles = read_jsonl_and_map_to_df(wiki_path, ["text"])
    filtered_articles = filter_articles(all_articles)
    article_texts = parse_article_text(filtered_articles)

    combined_tokens = []
    for article in article_texts.head(n = 15):
        all_tokens = tokenise_article(article)
        filtered_tokens = filter_tokens(all_tokens)
        print(filtered_tokens)
        #combined_tokens.extend(tokens)



    #print(filtered_articles.head())









#####################################



def get_word_counts(words: list, lower: bool = True, sort: bool = True) -> list:
    if lower:
        words = list(map(lambda x: str(x).lower(), words))
    counter = Counter(words)

    print("Counted word frequencies for {:,} words ({:,} unique).".format(len(words), len(counter.keys())))
    if sort:
        return counter.most_common()
    return list(counter.items())

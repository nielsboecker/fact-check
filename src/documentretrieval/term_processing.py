import re

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_article(article: str) -> str:
    return article.replace('-LRB-', '').replace('-RRB-', '')


def tokenise_article(article: str) -> list:
    return re.split(r'\s+|-', article)


def filter_tokens(tokens: list) -> list:
    # Filter words that are too short, not consisting of alphanumeric characters, or are stopwords
    regex = re.compile(r'^[a-zA-Z0-9]{2,}$')
    filtered_tokens = filter(regex.search, tokens)
    filtered_tokens = [word for word in filtered_tokens if word not in stop_words]
    return list(filtered_tokens)


def normalise_article(article):
    return article.lower()


def process_normalise_tokenise_filter(raw_article: str) -> list:
    article = preprocess_article(raw_article)
    normalised_article = normalise_article(article)
    all_tokens = tokenise_article(normalised_article)
    return filter_tokens(all_tokens)

import re

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def DEPRECATED_preprocess_article(article: str) -> str:
    return article.replace('-LRB-', '').replace('-RRB-', '')


def preprocess_doc_text(text: str) -> str:
    return re.sub(r'(-LRB-|-RRB-|-LSB-|-RSB-|-COLON-) ', '', text)


def recreate_punctuation_in_doc_text(text: str) -> str:
    return text \
        .replace('-LRB- ', '(') \
        .replace(' -RRB-', ')') \
        .replace('-LSB- ', ']') \
        .replace(' -RSB-', ']') \
        .replace(' -COLON-', ';')


def tokenise_doc_text(text: str) -> list:
    return re.split(r'\s+|-', text)


def filter_tokens(tokens: list) -> list:
    # Filter words that are too short, not consisting of alphanumeric characters, or are stopwords
    regex = re.compile(r'^[a-zA-Z0-9]{2,}$')
    filtered_tokens = filter(regex.search, tokens)
    filtered_tokens = [word for word in filtered_tokens if word not in stop_words]
    return list(filtered_tokens)


def normalise_doc_text(text: str):
    return text.lower()


def process_normalise_tokenise_filter(raw_text: str) -> list:
    preprocessed_text = preprocess_doc_text(raw_text)
    normalised_article = normalise_doc_text(preprocessed_text)
    all_tokens = tokenise_doc_text(normalised_article)
    return filter_tokens(all_tokens)

import pandas as pd

from documentretrieval.claim_processing import preprocess_text
from documentretrieval.term_processing import process_normalise_tokenise_filter


def filter_documents(articles: pd.DataFrame) -> pd.DataFrame:
    min_article_length = 20
    is_long_enough = articles['text'].str.len() > min_article_length
    filtered = articles[is_long_enough]
    print('Using {} articles after filtering'.format(len(filtered)))
    return filtered


def reduce_document_to_text_column(articles: pd.DataFrame) -> pd.DataFrame:
    return articles['text']


def preprocess_doc_title(page_id: str) -> list:
    doc_title_preprocessed = preprocess_text(page_id).replace('_', ' ')
    return process_normalise_tokenise_filter(doc_title_preprocessed)

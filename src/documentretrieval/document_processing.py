import pandas as pd


def filter_documents(articles: pd.DataFrame) -> pd.DataFrame:
    min_article_length = 20
    is_long_enough = articles['text'].str.len() > min_article_length
    filtered = articles[is_long_enough]
    print('Using {} articles after filtering'.format(len(filtered)))
    return filtered


def reduce_document_to_text_column(articles: pd.DataFrame) -> pd.DataFrame:
    return articles['text']

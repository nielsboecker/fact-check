import pandas as pd


def filter_articles(articles: pd.DataFrame) -> pd.DataFrame:
    min_article_length = 20
    is_long_enough = articles['text'].str.len() > min_article_length
    return articles[is_long_enough]


def parse_article_text(articles: pd.DataFrame) -> pd.DataFrame:
    return articles['text']
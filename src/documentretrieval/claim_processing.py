import re


def preprocess_claim(claim: str) -> str:
    # Add spaces around punctuation so that claims can be processed like wiki-pages
    return re.sub(r'([.,!?;])', r' \1 ', claim)

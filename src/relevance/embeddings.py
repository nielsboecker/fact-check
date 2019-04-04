import numpy as np

from dataaccess.access_glove_embeddings import get_embedding
from documentretrieval.claim_processing import preprocess_claim_text
from documentretrieval.term_processing import preprocess_doc_text
from util.vector_algebra import get_min_max_vectors


def transform_LR_input(claim_text: str, line_text: str, debug: bool = False):
    # remove punctuation that are otherwise part of tokens
    preprocessed_claim = preprocess_claim_text(claim_text)
    # remove artifacts like -LRB- etc.
    preprocessed_line = preprocess_doc_text(line_text)

    claim_vector = transform_sentence_to_vector(preprocessed_claim, debug)
    line_vector = transform_sentence_to_vector(preprocessed_line, debug)

    return create_LR_feature_vector(claim_vector, line_vector)


def create_LR_feature_vector(claim_vector: np.array, line_vector: np.array) -> np.array:
    return claim_vector - line_vector


def transform_sentence_to_vector(sentence: str, debug: bool = False):
    # Refer to https://arxiv.org/pdf/1607.00570.pdf
    embeddings = [get_embedding(term, debug) for term in sentence.split()]
    min_values, max_values = get_min_max_vectors(embeddings)
    sentence_vector = np.concatenate((min_values, max_values), axis=0)
    assert sentence_vector.shape == (600,)
    return sentence_vector

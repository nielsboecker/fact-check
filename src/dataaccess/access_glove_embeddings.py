import time

from gensim.models.keyedvectors import KeyedVectors

from dataaccess.constants import DATA_PRETRAINED_EMBEDDINGS_PATH

glove_model = None


def get_glove_model():
    global glove_model
    if not glove_model:
        start_time = time.time()
        print('Loading GloVe model into memory...')
        glove_model = KeyedVectors.load_word2vec_format(DATA_PRETRAINED_EMBEDDINGS_PATH, binary=False, limit=500)
        print('Loaded GloVe model in {:.2f} seconds'.format(time.time() - start_time))

    return glove_model

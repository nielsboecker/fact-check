import pyhash

hasher = pyhash.super_fast_hash()

# constants controlling amount of subindices
NUM_OF_INVERTED_INDEX_SHARDS = 10000

# paths to data
DATA_BASE_PATH = './data/'
DATA_WIKI_PATH = DATA_BASE_PATH + 'wiki-pages/'
DATA_TRAINING_PATH = DATA_BASE_PATH + 'train.jsonl'
DATA_DEV_LABELED_PATH = DATA_BASE_PATH + 'shared_task_dev.jsonl'
#DATA_PRETRAINED_EMBEDDINGS_PATH = DATA_BASE_PATH + 'GoogleNews-vectors-negative300.bin'
DATA_PRETRAINED_EMBEDDINGS_PATH = DATA_BASE_PATH + 'glove.840B.300d.txt'

# paths to generated auxiliary data and output
GENERATED_BASE_PATH = './generated/'
GENERATED_COUNTS_PATH = GENERATED_BASE_PATH + 'accumulated_word_count.jsonl'
GENERATED_IDF_PATH = GENERATED_BASE_PATH + 'words_with_idf.jsonl'
GENERATED_WIKI_PAGE_MAPPINGS_PATH = GENERATED_BASE_PATH + 'wiki_page_batch_mappings.p'
GENERATED_DOCUMENT_NORMS_MAPPING = GENERATED_BASE_PATH + 'docs_to_norms_mapping.jsonl'
GENERATED_DOCUMENT_LENGTH_MAPPING = GENERATED_BASE_PATH + 'docs_to_lengths_mapping.jsonl'
GENERATED_INVERTED_INDEX_DIRECTORY = GENERATED_BASE_PATH + 'inverted_index/'
GENERATED_PREPROCESSED_TRAINING_DATA = GENERATED_BASE_PATH + 'preprocessed_training_data.p'
GENERATED_PREPROCESSED_DEV_DATA = GENERATED_BASE_PATH + 'preprocessed_dev_data.p'
GENERATED_LOGISTIC_REGRESSION_MODEL = GENERATED_BASE_PATH + 'logistic_regression.p'
GENERATED_LOGISTIC_REGRESSION_LOSS_HISTORY = GENERATED_BASE_PATH + 'logistic_regression_loss.p'
GENERATED_FIGURES_BASE_PATH = GENERATED_BASE_PATH + 'figures/'

# Retrieved documents
RETRIEVED_BASE_PATH = './retrieved/'
RETRIEVED_TFIDF_DIRECTORY = RETRIEVED_BASE_PATH + 'tfidf/'
RETRIEVED_PROBABILISTIC_DIRECTORY = RETRIEVED_BASE_PATH + 'probabilistic/'

# Retrieval parameters
DOCS_TO_RETRIEVE_PER_CLAIM = 5

def get_wiki_batch_path(batch_id):
    return '{}wiki-{:03}.jsonl'.format(DATA_WIKI_PATH, batch_id)


def get_inverted_index_shard_id(page_id: str) -> int:
    return hasher(page_id) % NUM_OF_INVERTED_INDEX_SHARDS


def get_shard_path(shard_id: int):
    return '{}{:04}.json'.format(GENERATED_INVERTED_INDEX_DIRECTORY, shard_id)

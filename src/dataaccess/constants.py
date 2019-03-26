import pyhash

hasher = pyhash.super_fast_hash()

# constants controlling amount of subindices
NUM_OF_WIKI_ID_TO_BATCH_SUBMAPS = 1000
NUM_OF_INVERTED_INDEX_SHARDS = 10000

# collection-specific data
COLLECTION_DOCUMENTS_NUMBER = 5391645   # number of documents after filtering too short wiki-pages
COLLECTION_VOCABULARY_SIZE = 2697407    # unique terms after filtering wiki-pages and processing documents
COLLECTION_TOTAL_WORDS = 271036237      # number of total words after filtering wiki-pages and processing documents

# claims-specific data
CLAIMS_COLUMNS_LABELED = ['id', 'verifiable', 'label', 'claim', 'evidence']

# paths to data
DATA_BASE_PATH = './data/'
DATA_WIKI_PATH = DATA_BASE_PATH + 'wiki-pages/'
DATA_TRAINING_PATH = DATA_BASE_PATH + 'train.jsonl'
DATA_DEV_UNLABELED_PATH = DATA_BASE_PATH + 'shared_task_dev_public.jsonl'  # TODO: Wrong dataset
DATA_DEV_LABELED_PATH = DATA_BASE_PATH + 'shared_task_dev.jsonl'

# paths to generated interim data and output
GENERATED_BASE_PATH = './generated/'
GENERATED_COUNTS_PATH = GENERATED_BASE_PATH + 'accumulated_word_count.jsonl'
GENERATED_IDF_PATH = GENERATED_BASE_PATH + 'words_with_idf.jsonl'
GENERATED_WIKI_PAGE_SUBMAP_DIRECTORY = GENERATED_BASE_PATH + 'wiki_page_batch_mappings/'
GENERATED_DOCUMENT_NORMS_MAPPING = GENERATED_BASE_PATH + 'docs_to_norms_mapping.jsonl'
GENERATED_DOCUMENT_LENGTH_MAPPING = GENERATED_BASE_PATH + 'docs_to_lengths_mapping.jsonl'
GENERATED_INVERTED_INDEX_DIRECTORY = GENERATED_BASE_PATH + 'inverted_index/'

# Retrieved documents
RETRIEVED_BASE_PATH = './retrieved/'
RETRIEVED_TFIDF_DIRECTORY = RETRIEVED_BASE_PATH + 'tfidf/'
RETRIEVED_PROBABILISTIC_DIRECTORY = RETRIEVED_BASE_PATH + 'probabilistic/'

# Retrieval parameters
DOCS_TO_RETRIEVE_PER_CLAIM = 5

# 3rd party libs
TERM_COLOURS = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']


def get_wiki_batch_path(batch_id):
    return '{}wiki-{:03}.jsonl'.format(DATA_WIKI_PATH, batch_id)


def get_wiki_id_to_batch_submap_id(page_id: str) -> int:
    return hasher(page_id) % NUM_OF_WIKI_ID_TO_BATCH_SUBMAPS


def get_wiki_id_to_batch_submap_path(submap_id: int) -> str:
    return '{}{:03}.jsonl'.format(GENERATED_WIKI_PAGE_SUBMAP_DIRECTORY, submap_id)


def get_inverted_index_shard_id(page_id: str) -> int:
    return hasher(page_id) % NUM_OF_INVERTED_INDEX_SHARDS


def get_shard_path(shard_id: int):
    return '{}{:04}.json'.format(GENERATED_INVERTED_INDEX_DIRECTORY, shard_id)

import pyhash

# DATA_COLUMNS_UNLABELED = ['id', 'claim']
CLAIMS_COLUMNS_LABELED = ['id', 'verifiable', 'label', 'claim', 'evidence']
NUM_OF_WIKI_ID_TO_BATCH_SUBMAPS = 1000
NUM_OF_INVERTED_INDEX_SHARDS = 2500

hasher = pyhash.super_fast_hash()

# paths to data
DATA_BASE_PATH = './data/'
DATA_WIKI_PATH = DATA_BASE_PATH + 'wiki-pages/'
DATA_EXAMPLE_PATH = DATA_BASE_PATH + 'example.jsonl'
DATA_TRAINING_PATH = DATA_BASE_PATH + 'train.jsonl'
DATA_DEV_UNLABELED_PATH = DATA_BASE_PATH + 'shared_task_dev_public.jsonl'  # TODO: Wrong dataset
DATA_DEV_LABELED_PATH = DATA_BASE_PATH + 'shared_task_dev.jsonl'

# paths to generated interim data and output
GENERATED_BASE_PATH = './generated/'
GENERATED_COUNTS_PATH = GENERATED_BASE_PATH + 'accumulated_word_count.jsonl'
GENERATED_IDF_PATH = GENERATED_BASE_PATH + 'words_with_idf.jsonl'
GENERATED_WIKI_PAGE_SUBMAP_DIRECTORY = GENERATED_BASE_PATH + 'wiki-page-batch-mappings/'
GENERATED_INVERTED_INDEX_DIRECTORY = GENERATED_BASE_PATH + 'inverted-index/'

# Retrieved documents
RETRIEVED_BASE_PATH = './retrieved/'
RETRIEVED_TFIDF_DIRECTORY = RETRIEVED_BASE_PATH + 'tf-idf/'


def get_wiki_batch_path(batch_id):
    return '{}wiki-{:03}.jsonl'.format(DATA_WIKI_PATH, batch_id)


def get_wiki_id_to_batch_submap_id(page_id: str) -> int:
    return hasher(page_id) % NUM_OF_WIKI_ID_TO_BATCH_SUBMAPS


def get_wiki_id_to_batch_submap_path(submap_id: int) -> str:
    return '{}{:03}.jsonl'.format(GENERATED_WIKI_PAGE_SUBMAP_DIRECTORY, submap_id)


def get_inverted_index_shard_id(page_id: str) -> int:
    return hasher(page_id) % NUM_OF_INVERTED_INDEX_SHARDS


def get_shard_path(shard_id: int):
    return '{}{}.json'.format(GENERATED_INVERTED_INDEX_DIRECTORY, shard_id)
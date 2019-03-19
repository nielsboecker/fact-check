# paths to data
DATA_BASE_PATH = './data/'
DATA_WIKI_PATH = DATA_BASE_PATH + 'wiki-pages/'
DATA_EXAMPLE_PATH = DATA_BASE_PATH + 'example.jsonl'
DATA_TRAINING_PATH = DATA_BASE_PATH + 'train.jsonl'
DATA_DEV_UNLABELED_PATH = DATA_BASE_PATH + 'shared_task_dev_public.jsonl' # TODO: Wrong dataset
DATA_DEV_LABELED_PATH = DATA_BASE_PATH + 'shared_task_dev.jsonl'

# paths to generated interim data and output
GENERATED_BASE_PATH = './generated/'
GENERATED_COUNTS_PATH = GENERATED_BASE_PATH + 'accumulated_word_count.jsonl'
GENERATED_IDF_PATH = GENERATED_BASE_PATH + 'words_with_idf.jsonl'
GENERATED_WIKI_PAGE_SUBMAP_DIRECTORY = GENERATED_BASE_PATH + 'wiki-page-batch-mappings/'
GENERATED_INVERTED_INDEX_DIRECTORY = GENERATED_BASE_PATH + 'inverted-index/'

# Retrieved documents
RETRIEVED_BASE_PATH = './retrieved/'
RETRIEVED_TFIDF_DIRECTORY = RETRIEVED_BASE_PATH + 'tf-idf'

# DATA_COLUMNS_UNLABELED = ['id', 'claim']
# DATA_COLUMNS_LABELED = ['id', 'verifiable', 'label', 'claim', 'evidence']

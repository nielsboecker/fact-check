# paths to data
DATA_BASE_PATH = './data/'
DATA_WIKI_PATH = DATA_BASE_PATH + 'wiki-pages/'
DATA_EXAMPLE_PATH = DATA_BASE_PATH + 'example.jsonl'
DATA_TRAINING_PATH = DATA_BASE_PATH + 'train.jsonl'
DATA_DEV_UNLABELED_PATH = DATA_BASE_PATH + 'shared_task_dev_public.jsonl' # TODO: Wrong dataset
DATA_DEV_LABELED_PATH = DATA_BASE_PATH + 'shared_task_dev.jsonl'

# paths to generated output
GENERATED_BASE_PATH = './generated/'
GENERATED_COUNTS_PATH = GENERATED_BASE_PATH + 'accumulated_word_count.jsonl'
GENERATED_IDFS_PATH = GENERATED_BASE_PATH + 'words_and_idfs.jsonl'
GENERATED_WIKI_BATCHES_FIRST_ROW_MAPPING_PATH = GENERATED_BASE_PATH + 'wiki_batch_first_row_mapping.jsonl'

# DATA_COLUMNS_UNLABELED = ['id', 'claim']
# DATA_COLUMNS_LABELED = ['id', 'verifiable', 'label', 'claim', 'evidence']

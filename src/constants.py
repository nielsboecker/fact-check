# paths to data
DATA_BASE_PATH = './data/'
DATA_EXAMPLE_PATH = DATA_BASE_PATH + 'example.jsonl'
DATA_TRAINING_PATH = DATA_BASE_PATH + 'train.jsonl'
DATA_DEV_UNLABELED_PATH = DATA_BASE_PATH + 'shared_task_dev_public.jsonl'
DATA_DEV_LABELED_PATH = DATA_BASE_PATH + 'shared_task_dev.jsonl'

DATA_COLUMNS_UNLABELED = ['id', 'claim']
DATA_COLUMNS_LABELED = ['id', 'verifiable', 'label', 'claim', 'evidence']

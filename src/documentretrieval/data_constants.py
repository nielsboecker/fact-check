# collection-specific data
COLLECTION_DOCUMENTS_NUMBER = 5391645   # number of documents after filtering too short wiki-pages
COLLECTION_VOCABULARY_SIZE = 2697407    # unique terms after filtering wiki-pages and processing documents
COLLECTION_TOTAL_WORDS = 271036237      # number of total words after filtering wiki-pages and processing documents

# claims-specific data
CLAIMS_COLUMNS_LABELED = ['id', 'verifiable', 'label', 'claim', 'evidence']
CLAIMS_COLUMNS_UNLABELED = ['id', 'claim']

# intermediate data format
PREPROCESSED_DATA_COLUMNS_V1 = ['claim_id', 'page_id', 'line_id', 'input_vector', 'expected_output']
PREPROCESSED_DATA_COLUMNS_V2 = ['claim_id', 'input_vector', 'expected_output']

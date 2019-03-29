from dataaccess.constants import GENERATED_DOCUMENT_LENGTH_MAPPING
from dataaccess.files_io import read_jsonl_and_map_to_df

doc_length_mapping = read_jsonl_and_map_to_df(GENERATED_DOCUMENT_LENGTH_MAPPING, ['page_id', 'length']).set_index('page_id', drop=False)

def get_length_of_doc(page_id: str) -> int:
    return doc_length_mapping.loc[page_id]['length']

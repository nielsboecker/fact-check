from dataaccess.constants import GENERATED_DOCUMENT_NORMS_MAPPING
from dataaccess.json_io import read_jsonl_and_map_to_df

docs_norms = read_jsonl_and_map_to_df(GENERATED_DOCUMENT_NORMS_MAPPING, ['doc', 'norm']).set_index('doc', drop=False)


def get_norm_for_doc(page_id: str) -> float:
    return docs_norms.loc[page_id]['norm']

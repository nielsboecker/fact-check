from dataaccess.constants import get_inverted_index_shard_id, get_shard_path
from dataaccess.json_io import read_dict_from_json


def read_inverted_index_shard(shard_id: int) -> dict:
    shard_path = get_shard_path(shard_id)
    shard = read_dict_from_json(shard_path)
    return shard


def get_index_entry_for_term(term: str) -> dict:
    shard_id = get_inverted_index_shard_id(term)
    shard = read_inverted_index_shard(shard_id)
    return shard[term]


def get_candidate_documents_for_claim(claim_terms: list, mode: str = 'tfidf') -> dict:
    doc_candidates = {}
    # In the index, for each term, the occurrences per document are stored
    # For all terms, group by document to compute TF values
    for term in claim_terms:
        index_entry = get_index_entry_for_term(term)
        docs = index_entry['docs']
        for doc in docs:
            page_id = doc[0]
            raw_count = doc[1]
            tfidf_for_term = doc[2]
            value = tfidf_for_term if mode == 'tfidf' else raw_count
            doc_candidates.setdefault(page_id, {})[term] = value

    print('Found {:,} documents containing one or more claim terms ({}")'.format(len(doc_candidates), claim_terms))
    return doc_candidates
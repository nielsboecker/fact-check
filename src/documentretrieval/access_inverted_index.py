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

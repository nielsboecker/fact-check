import time

import pyhash
from _1_A_word_frequency_count import get_wiki_batch_path
from constants import GENERATED_WIKI_PAGE_SUBMAP_DIRECTORY
from jsonl_io import read_jsonl_and_map_to_df, write_dict_to_json

NUM_OF_SUBMAPS = 1000
hasher = pyhash.super_fast_hash()


def get_submap_id(page_id: str) -> int:
    return hasher(page_id) % NUM_OF_SUBMAPS



def get_submap_path(submap_id: int) -> str:
    return '{}{:03}.jsonl'.format(GENERATED_WIKI_PAGE_SUBMAP_DIRECTORY, submap_id)


def generate_submaps():
    submaps = {}
    for i in range(NUM_OF_SUBMAPS):
        submaps[i] = {}

    for batch_id in range(1, 110):
        print('Processing batch #{}...'.format(batch_id))
        batch_file_path = get_wiki_batch_path(batch_id)
        batch_df = read_jsonl_and_map_to_df(batch_file_path, ['id'])
        for line_index, row in batch_df.iterrows():
            page_id = row[0]
            submap_id = get_submap_id(page_id)
            submaps[submap_id][page_id] = (batch_id, line_index)

    print('Storing submaps on disk...')
    for entry in submaps.items():
        submap_id = entry[0]
        submap_mappings = entry[1]
        submap_file_path = get_submap_path(submap_id)
        write_dict_to_json(submap_file_path, submap_mappings)


if __name__ == '__main__':
    start_time = time.time()
    generate_submaps()
    print('Finished submap generation after {:.2f} seconds'.format(time.time() - start_time))

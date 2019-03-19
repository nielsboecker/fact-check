import time
from multiprocessing import Pool, cpu_count

import pyhash
from termcolor import colored

from dataaccess.constants import get_wiki_batch_path, NUM_OF_WIKI_ID_TO_BATCH_SUBMAPS, get_wiki_id_to_batch_submap_id, \
    get_wiki_id_to_batch_submap_path
from dataaccess.json_io import read_jsonl_and_map_to_df, write_dict_to_json

hasher = pyhash.super_fast_hash()


def generate_batch_submaps(batch_id: int):
    print('Processing batch #{}...'.format(batch_id))
    parital_submap = {}
    for i in range(NUM_OF_WIKI_ID_TO_BATCH_SUBMAPS):
        parital_submap[i] = {}

    batch_file_path = get_wiki_batch_path(batch_id)
    batch_df = read_jsonl_and_map_to_df(batch_file_path, ['id'])
    for line_index, row in batch_df.iterrows():
        page_id = row[0]
        submap_id = get_wiki_id_to_batch_submap_id(page_id)
        parital_submap[submap_id][page_id] = (batch_id, line_index)

    return parital_submap


def generate_all_submaps():
    num_processes = cpu_count()  # max(cpu_count() - 2, 2)
    print(colored('Detected {} CPUs, spawning {} processes'.format(cpu_count(), num_processes), attrs=['bold']))
    pool = Pool(processes=num_processes)

    start_index_inclusive = 1
    stop_index_exclusive = 110
    # Process in multiple blocking processes
    partial_submaps = pool.map(generate_batch_submaps, range(start_index_inclusive, stop_index_exclusive))
    pool.close()

    print('Merging {} partial results...'.format(len(partial_submaps)))
    accumuated_submaps = {}
    for i in range(NUM_OF_WIKI_ID_TO_BATCH_SUBMAPS):
        accumuated_submaps[i] = {}

    # Go through all processes' partial results ...
    for result in partial_submaps:
        # ... and for all submap hashes/IDs, merge them into accumulated result
        for i in range(NUM_OF_WIKI_ID_TO_BATCH_SUBMAPS):
            accumuated_submaps[i].update(result[i])

    # Store result on disk
    print('Storing submaps on disk...')
    for entry in accumuated_submaps.items():
        submap_id = entry[0]
        submap_mappings = entry[1]
        submap_file_path = get_wiki_id_to_batch_submap_path(submap_id)
        write_dict_to_json(submap_file_path, submap_mappings)


if __name__ == '__main__':
    start_time = time.time()
    generate_all_submaps()
    print('Finished submap generation after {:.2f} seconds'.format(time.time() - start_time))

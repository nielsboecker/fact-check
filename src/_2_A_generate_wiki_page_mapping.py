import argparse
import time

import pandas as pd

from dataaccess.files_constants import get_wiki_batch_path, GENERATED_WIKI_PAGE_MAPPINGS_PATH
from dataaccess.files_io import read_jsonl_and_map_to_df, write_pickle
from util.theads_processes import get_process_pool

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='use subset of data', action='store_true')
args = parser.parse_args()

def generate_batch_mappings(batch_id: int):
    print('Processing batch #{}...'.format(batch_id))
    parital_result = {}

    batch_file_path = get_wiki_batch_path(batch_id)
    batch_df = read_jsonl_and_map_to_df(batch_file_path, ['id'])
    for line_index, row in batch_df.iterrows():
        page_id = row[0]
        parital_result[page_id] = (batch_id, line_index)

    return parital_result


def generate_all_mappings():
    pool = get_process_pool()

    start_index_inclusive = 1 if not args.debug else 109
    stop_index_exclusive = 110
    partial_mappings = pool.map(generate_batch_mappings, range(start_index_inclusive, stop_index_exclusive))
    pool.close()

    print('Merging {} partial results...'.format(len(partial_mappings)))
    accumulated_mappings = {}

    for partial_result in partial_mappings:
            accumulated_mappings.update(partial_result)

    mapping = pd.DataFrame.from_dict(accumulated_mappings, orient='index', columns=['batch_id', 'line'])
    print(mapping.head())
    write_pickle(GENERATED_WIKI_PAGE_MAPPINGS_PATH, mapping)


if __name__ == '__main__':
    start_time = time.time()
    generate_all_mappings()
    print('Finished submap generation after {:.2f} seconds'.format(time.time() - start_time))

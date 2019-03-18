import argparse

from constants import DATA_WIKI_PATH, GENERATED_WIKI_BATCHES_FIRST_ROW_MAPPING_PATH
from jsonl_io import read_jsonl_and_map_to_dict


parser = argparse.ArgumentParser()
parser.add_argument("--test", help="retrieve a document to test module", action="store_true")
args = parser.parse_args()


batch_first_row_map = read_jsonl_and_map_to_dict(GENERATED_WIKI_BATCHES_FIRST_ROW_MAPPING_PATH)
reversed_batch_first_row_map = [entry for entry in batch_first_row_map.items()]
reversed_batch_first_row_map.reverse()


def get_batch_id(page_id: str) -> int:
    # First, find right batch by going backwards through the first words of batches
    for entry in reversed_batch_first_row_map:
        if page_id > entry[1]:
            return entry[0]
        else:
            print('Not {}'.format(entry[0]))
    print('Error: Couldn\'t find batch for page_id ""'.format(page_id))


def retrieve_wiki_page(page_id: str) -> dict:
    print(get_batch_id(page_id))
    return 'TODO'


if __name__ == '__main__':
    if (args.test):
        # Habrosaurus, random wiki-page in batch #42
        print(retrieve_wiki_page('Habrosaurus'))
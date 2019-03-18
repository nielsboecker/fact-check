import json

from constants import DATA_WIKI_PATH, GENERATED_WIKI_BATCHES_FIRST_ROW_MAPPING_PATH
from jsonl_io import write_list_to_jsonl


def generate_batches_first_row_mapping():
    first_row_mapping = {}
    for batch_id in range(1, 110):
        batch_file_path = '{}wiki-{:03}.jsonl'.format(DATA_WIKI_PATH, batch_id)
        with open(batch_file_path) as fp:
            for line in fp:
                first_row = json.loads(line)
                first_row_mapping[batch_id] = first_row['id']
                # Only look at first line of each file
                break
    assert(len(first_row_mapping) == 109)

    mapping_entries = [entry for entry in first_row_mapping.items()]
    write_list_to_jsonl(GENERATED_WIKI_BATCHES_FIRST_ROW_MAPPING_PATH, mapping_entries)


if __name__ == '__main__':
    generate_batches_first_row_mapping()
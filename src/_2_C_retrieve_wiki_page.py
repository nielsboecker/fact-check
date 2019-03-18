import argparse
import time

from _1_A_word_frequency_count import get_wiki_batch_path
from _2_A_generate_wiki_page_mapping import get_submap_id, get_submap_path
from json_io import read_dict_from_json
from termcolor import colored
from wiki_document import WikiDocument

parser = argparse.ArgumentParser()
parser.add_argument("--id", help="ID of a document to retrieve for test purposes")#, action="store_true")
args = parser.parse_args()



def retrieve_wiki_page(page_id: str) -> dict:
    # Use supmap id to retrieve the actual mapping
    submap_id = get_submap_id(page_id)
    submap_path = get_submap_path(submap_id)
    submap = read_dict_from_json(submap_path)

    # Find actual mapping and read only relevant line from wiki-pages batch
    batch_id, line_index = submap[page_id]
    wiki_batch_path = get_wiki_batch_path(batch_id)

    with open(wiki_batch_path) as fp:
        for i, line in enumerate(fp):
            if i == line_index:
                return line

    # If this code runs, a mapping error occured
    print(colored('Coudn\'t find document "{}" in batch #{}, after mapping from submap {}'.format(page_id, batch_id, submap_id)), 'red')


if __name__ == '__main__':
    if (args.id):
        start_time = time.time()
        wiki_page_string = retrieve_wiki_page(args.id)
        wiki_document = WikiDocument(wiki_page_string)
        print('Retrieved document "{}" after {:.5f} seconds'.format(args.id, time.time() - start_time))
        print(colored(wiki_document.text, 'blue'))
    else:
        print('Please add ID to retrieve')
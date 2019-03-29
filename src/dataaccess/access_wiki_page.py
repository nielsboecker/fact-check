from termcolor import colored

from dataaccess.constants import get_wiki_id_to_batch_submap_id, get_wiki_id_to_batch_submap_path, get_wiki_batch_path
from dataaccess.files_io import read_dict_from_json
from model.wiki_document import WikiDocument


def retrieve_wiki_page(page_id: str) -> WikiDocument:
    # Use supmap id to retrieve the actual mapping
    submap_id = get_wiki_id_to_batch_submap_id(page_id)
    submap_path = get_wiki_id_to_batch_submap_path(submap_id)
    submap = read_dict_from_json(submap_path)

    # Find actual mapping and read only relevant line from wiki-pages batch
    batch_id, line_index = submap[page_id]
    wiki_batch_path = get_wiki_batch_path(batch_id)

    with open(wiki_batch_path) as fp:
        for i, line in enumerate(fp):
            if i == line_index:
                return WikiDocument(line)

    # If this code runs, a mapping error occured
    print(colored('Coudn\'t find document "{}" in batch #{}, after mapping from submap {}'.format(page_id, batch_id, submap_id)), 'red')

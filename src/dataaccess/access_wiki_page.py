import pandas as pd
from termcolor import colored

from dataaccess.files_constants import get_wiki_batch_path, GENERATED_WIKI_PAGE_MAPPINGS_PATH
from dataaccess.files_io import read_pickle
from model.wiki_document import WikiDocument

wiki_page_mapping: pd.DataFrame = read_pickle(GENERATED_WIKI_PAGE_MAPPINGS_PATH)


def retrieve_wiki_page(page_id: str) -> WikiDocument:
    # Find correct batch file and read only relevant line
    batch_id, line = wiki_page_mapping.loc[page_id].values
    wiki_batch_path = get_wiki_batch_path(batch_id)

    with open(wiki_batch_path) as fp:
        for i, json_line in enumerate(fp):
            if i == line:
                return WikiDocument(json_line)

    # If this code runs, a mapping error occured
    print(colored('Error: Line {} not found in wiki-page {}'.format(line, batch_id), 'red'))

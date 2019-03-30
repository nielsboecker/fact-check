import re

from termcolor import colored

from dataaccess.access_wiki_page import retrieve_wiki_page
from dataaccess.files_io import write_list_to_oneline_csv


def preprocess_text(claim: str) -> str:
    # Add spaces around punctuation so that claims can be further processed like the text in wiki-pages
    return re.sub(r'([.,!?;])', r' \1 ', claim)


def display_or_store_result(claim: str, claim_id: int, result_docs: list, dir_path: str, display_only: bool = False):
    if display_only:
        print(colored('Results for claim "{}":'.format(claim), 'yellow'))
        for doc in result_docs:
            page_id = doc[0]
            wiki_page = retrieve_wiki_page(page_id)
            print(wiki_page)
    else:
        #result_path = '{}{}.jsonl'.format(path, claim_id)
        #write_list_to_jsonl(result_path, result_docs)
        print(colored('Storing results for claim "{}"\n{}:'.format(claim, result_docs), 'yellow'))
        write_list_to_oneline_csv(dir_path, claim_id, result_docs)

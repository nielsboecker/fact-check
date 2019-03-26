import re

from termcolor import colored

from _2_F_retrieve_docs_with_tfidf import args
from dataaccess.access_wiki_page import retrieve_wiki_page
from dataaccess.json_io import write_list_to_jsonl


def preprocess_claim(claim: str) -> str:
    # Add spaces around punctuation so that claims can be processed like wiki-pages
    return re.sub(r'([.,!?;])', r' \1 ', claim)


def display_or_store_result(claim: str, claim_id: int, result_docs: list, path: str):
    if (args.print):
        print(colored('Results for claim "{}":'.format(claim), attrs=['bold']))
        for doc in result_docs:
            page_id = doc[0]
            wiki_page = retrieve_wiki_page(page_id)
            print(wiki_page)
    else:
        result_path = '{}{}.jsonl'.format(path, claim_id)
        write_list_to_jsonl(result_path, result_docs)

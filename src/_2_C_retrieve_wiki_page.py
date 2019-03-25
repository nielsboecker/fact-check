import argparse
import time

from dataaccess.access_wiki_page import retrieve_wiki_page

parser = argparse.ArgumentParser()
parser.add_argument("--id", help="ID of a document to retrieve for test purposes")
args = parser.parse_args()

if __name__ == '__main__':
    if (args.id):
        start_time = time.time()
        wiki_document = retrieve_wiki_page(args.id)
        print('Retrieved document "{}" after {:.5f} seconds'.format(args.id, time.time() - start_time))
        print(wiki_document)
    else:
        print('Please add ID to retrieve')

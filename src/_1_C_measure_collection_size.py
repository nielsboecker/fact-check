import time

from termcolor import colored

from dataaccess.access_terms_frequencies_mapping import get_terms_with_occurrences_mapping
from dataaccess.constants import get_wiki_batch_path
from dataaccess.json_io import read_jsonl_and_map_to_df
from documentretrieval.document_processing import filter_documents


def count_documents_batch(batch_id: int) -> int:
    batch_file_path = get_wiki_batch_path(batch_id)
    all_articles = read_jsonl_and_map_to_df(batch_file_path, ['text'])
    filtered_articles = filter_documents(all_articles)
    return len(filtered_articles)


def count_documents_all() -> int:
    start_index_inclusive = 1
    stop_index_exclusive = 110
    accumulated_collection_size = 0
    for id in range(start_index_inclusive, stop_index_exclusive):
        accumulated_collection_size += count_documents_batch(id)
    return accumulated_collection_size


def count_words_all() -> int:
    return get_terms_with_occurrences_mapping()['frequency'].sum()


if __name__ == '__main__':
    start_time = time.time()
    collection_documents = count_documents_all()
    print(colored('After filtering, collection consists of {} documents.'.format(collection_documents)))

    collection_words = count_words_all()
    print(colored('After filtering, collection consists of {:,} words.'.format(collection_words)))

    print('Finished processing after {:.2f} seconds'.format(time.time() - start_time))

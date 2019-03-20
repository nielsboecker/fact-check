import argparse
import time
from collections import Counter
from multiprocessing import cpu_count, Pool

from dataaccess.constants import get_wiki_batch_path, NUM_OF_INVERTED_INDEX_SHARDS, get_inverted_index_shard_id, \
    get_shard_path, GENERATED_IDF_PATH
from dataaccess.json_io import read_jsonl_and_map_to_df, write_dict_to_json
from documentretrieval.document_processing import filter_articles
from documentretrieval.term_processing import process_normalise_tokenise_filter

parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="only use subset of data", action="store_true")
args = parser.parse_args()

words_with_idf = read_jsonl_and_map_to_df(GENERATED_IDF_PATH, ['word', 'idf']).set_index('word', drop=False)


def init_index_entry_for_term(term: str):
    index_entry = {}
    index_entry['idf'] = words_with_idf.loc[term]['idf']
    index_entry['docs'] = []
    return index_entry


def generate_partial_subindex_for_batch(batch_id: int) -> dict:
    batch_file_path = get_wiki_batch_path(batch_id)
    all_articles = read_jsonl_and_map_to_df(batch_file_path, ['id', 'text'])
    filtered_wiki_pages = filter_articles(all_articles)
    print('Using {} articles after filtering'.format(len(filtered_wiki_pages)))

    subindex = {}
    for raw_article in filtered_wiki_pages.iterrows():
        page_id = raw_article[1]['id']
        filtered_tokens = process_normalise_tokenise_filter(raw_article[1]['text'])
        words_counter = Counter(filtered_tokens)
        # Invert word -> doc and add raw and relative term count
        for count in words_counter.items():
            term = count[0]
            raw_count = count[1]
            relative_count = raw_count / len(filtered_tokens)
            subindex.setdefault(term, { 'docs': [] })['docs'].append((page_id, raw_count, relative_count))

    print('Finished processing batch #{}'.format(batch_id))
    return subindex


def store_shard(shard_id: int, shard: dict):
    shard_path = get_shard_path(shard_id)
    write_dict_to_json(shard_path, shard)


def generate_inverted_index_complete():
    print(('Detected {} CPUs'.format(cpu_count())))
    pool = Pool(processes=cpu_count())

    start_index_inclusive = 1
    stop_index_exclusive = 2 if args.debug else 110

    # Process in multiple blocking processes
    partial_subindices = pool.map(generate_partial_subindex_for_batch,
                                  range(start_index_inclusive, stop_index_exclusive))

    print('Merging {} partial results...'.format(len(partial_subindices)))
    inverted_index_shards = {} #{i: {} for i in range(NUM_OF_INVERTED_INDEX_SHARDS)}

    # For each partial result...
    for subindex in partial_subindices:
        # ...go through the entries for every word...
        for subindex_entry in subindex.items():
            # ...and merge with the corresponding entry in the corresponding shard
            term = subindex_entry[0]
            inner_entry = subindex_entry[1]
            docs = inner_entry['docs']
            shard_id = get_inverted_index_shard_id(term)
            shard = inverted_index_shards.setdefault(shard_id, {})
            shard.setdefault(term, init_index_entry_for_term(term))['docs'].extend(docs)

    # Store shards on disk
    print('Storing inverted index shards on disk...')
    pool.starmap(store_shard, inverted_index_shards.items())


if __name__ == '__main__':
    start_time = time.time()
    generate_inverted_index_complete()
    print('Finished index generation after {:.2f} seconds'.format(time.time() - start_time))

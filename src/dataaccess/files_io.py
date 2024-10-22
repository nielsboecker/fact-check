import csv
import json
import os
import pickle

import jsonlines
import pandas as pd
from termcolor import colored


def create_dir_if_not_exists(file_path: str):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        print('Creating {} directory'.format(dir_path))
        os.makedirs(dir_path, exist_ok=True)


def read_dict_from_json(input_path: str) -> dict:
    with open(input_path) as fp:
        data = json.load(fp)
        return data


def write_dict_to_json(output_path: str, data: dict):
    create_dir_if_not_exists(output_path)
    with open(output_path, 'w') as fp:
        json.dump(data, fp)


def read_jsonl_and_map_to_df(input_path: str, columns: list = None) -> pd.DataFrame:
    items = []
    with jsonlines.open(input_path) as reader:
        for fact in reader:
            items.append(fact)
    itemsDF = pd.DataFrame(items, columns=columns)
    print('Read {} lines from "{}"'.format(len(items), input_path))
    return itemsDF


def write_dict_to_jsonl(output_path: str, data: dict):
    write_list_to_jsonl(output_path, [i for i in data.items()])


def write_list_to_jsonl(output_path: str, data: list):
    create_dir_if_not_exists(output_path)
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(data)
    print(colored('Wrote {} lines to "{}"'.format(len(data), output_path), attrs=['bold']))


def write_dataframe_to_csv(output_path: str, data: pd.DataFrame, sep: str = ',', header: bool = True):
    data.to_csv(output_path, sep=sep, header=header, index=False)


def write_list_to_oneline_csv(dir_path: str, claim_id: int, list_of_tuples: list):
    create_dir_if_not_exists(dir_path)
    with open('{}{}.csv'.format(dir_path, claim_id), 'w', newline='') as fp:
        writer = csv.writer(fp, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([claim_id] + [tup[0] for tup in list_of_tuples])


def write_pickle(output_path: str, payload: object):
    pickle.dump(payload, open(output_path, "wb"))
    print('Stored pickle to "{}"'.format(output_path))


def read_pickle(input_path: str):
    loaded_pickle = pickle.load(open(input_path, "rb"))
    print('Loaded pickle from {}'.format(input_path))
    return loaded_pickle

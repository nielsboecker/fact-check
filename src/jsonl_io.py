import jsonlines
import pandas as pd
from termcolor import colored


def read_jsonl_and_map_to_df(input_path: str, columns: list) -> pd.DataFrame:
    items = []
    with jsonlines.open(input_path) as reader:
        for fact in reader:
            items.append(fact)
    itemsDF = pd.DataFrame(items, columns=columns)
    print(colored('Read {} lines from "{}"'.format(len(items), input_path), attrs=['bold']))
    return itemsDF


def write_list_to_jsonl(output_path: str, data: list):
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(data)
    print(colored('Wrote {} lines to "{}"'.format(len(data), output_path), attrs=['bold']))

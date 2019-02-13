import jsonlines
import pandas as pd

def read_jsonl_and_map_to_df(jsonl_path: str, columns: list) -> pd.DataFrame:
    items = []
    with jsonlines.open(jsonl_path) as reader:
        for fact in reader:
            items.append(fact)
    itemsDF = pd.DataFrame(items, columns=columns)
    print('Read {} lines from "{}".'.format(len(items), jsonl_path))
    return itemsDF

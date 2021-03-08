from typing import List, Union

import datasets
from datasets import load_dataset
import jsonlines

from common.qa_example import QAExample


def save(examples: List[QAExample], path: str):
    with jsonlines.open(path, 'w') as file:
        for example in examples:
            file.write(example.to_dict())


def load(paths: Union[str, List[str]]) -> datasets.Dataset:
    return load_dataset("json", data_files=paths, split="train")  # ignore 'split' argument

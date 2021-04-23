from typing import List, Union

import jsonlines

from training.example import Example
from training.test_example import TestExample


def save_examples(examples: List[Union[Example, TestExample]], path):
    with jsonlines.open(path, 'w') as writer:
        for example in examples:
            writer.write(example.to_dict())


def load_examples(path: Union[List[str], str]) -> List[Example]:
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path

    examples = []
    for p in paths:
        with jsonlines.open(p) as reader:
            for line in reader:
                example = Example(
                    question=line['question'],
                    context=line['context'],
                    answer=line['answer'],
                    answer_start_index=line['answer_start_index'],
                    is_impossible=line['is_impossible'],
                    category=line['category'],
                )
                examples.append(example)

    return examples


def load_test_examples(path: Union[List[str], str]) -> List[TestExample]:
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path

    examples = []
    for p in paths:
        with jsonlines.open(p) as reader:
            for line in reader:
                example = TestExample(
                    question=line['question'],
                    context=line['context'],
                    answers=[line['answer']],
                    category=line['category'],
                )
                examples.append(example)

    return examples


def read_number_of_samples(path: Union[List[str], str]) -> int:
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path

    n_samples = 0
    for p in paths:
        with jsonlines.open(p) as reader:
            for _ in reader:
                n_samples += 1
    return n_samples

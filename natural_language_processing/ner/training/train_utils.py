from typing import List, Tuple, Set

import jsonlines
import numpy as np

from src.ner.training.example import Example


class LabelEncoding:
    MAP = {
        'O': 0,
        'B-DS': 1,
        'I-DS': 2,
    }

    @classmethod
    def all_tags(cls) -> Set[str]:
        return set(LabelEncoding.MAP.keys())


def extract_candidates(prediction: List[int], last_valid: int):
    """Returns indices corresponding to entities.

        prediction: one of [0, 1, 2].
        last_valid: first index masked section.
    """
    indices = [
        i for i, p in enumerate(prediction) if p == 1
    ]  # extract starting indices

    if len(indices) == 0:
        return []

    candidates = []
    for index in indices:
        if index == 0:  # '101' token
            continue

        if index >= last_valid:
            continue

        end_index = None

        if index + 1 == last_valid:
            end_index = index + 1
        elif prediction[index + 1] == 0 or prediction[index + 1] == 1:
            end_index = index + 1
        elif prediction[index + 1] == 2:
            for i in range(index + 1, last_valid):
                if prediction[i] == 2:
                    end_index = i
                else:
                    break
                end_index = end_index + 1

        if end_index:
            candidates.append((index, end_index))

    return candidates


def extract_datasets(candidates: List[Tuple[int, int]], input_ids: List[int], tokenizer):
    datasets = []
    for start_index, end_index in candidates:
        datasets.append(
            tokenizer.decode(input_ids[start_index:end_index])
        )
    return datasets


def get_probabilities(candidates: List[Tuple[int, int]], prediction: np.ndarray) -> List[float]:
    probabilties = []
    for start_index, end_index in candidates:
        vector = prediction[start_index]
        p = np.exp(vector[1]) / np.exp(vector).sum()
        probabilties.append(p)
    return probabilties


def read_training_set(json_path) -> List[Example]:
    examples = []
    with jsonlines.open(json_path) as reader:
        for line in reader:
            examples.append(
                Example(
                    tokens=line['tokens'],
                    tags=line['tags']
                )
            )
    return examples

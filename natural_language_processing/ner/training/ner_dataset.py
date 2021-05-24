from typing import List

import torch
from torch.utils.data.dataset import Dataset

from src.ner.training.example import Example
from src.ner.training.train_utils import LabelEncoding


class NERDataset(Dataset):
    def __init__(self, examples: List[Example], tokenizer, max_len: int):
        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.n_tags = 3

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        input_ids = self.tokenizer.convert_tokens_to_ids(example.tokens)
        input_ids = input_ids[:self.max_len - 2]
        tags = example.tags[:self.max_len - 2]

        assert len(input_ids) == len(tags)

        input_ids = [101] + input_ids + [102]
        tags = [-100] + [LabelEncoding.MAP[tag] for tag in tags] + [-100]
        padding_len = self.max_len - len(input_ids)

        token_type_ids = [0] * len(input_ids) + [0] * padding_len
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        input_ids = input_ids + [0] * padding_len
        tags = tags + [0] * padding_len

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'tags': torch.tensor(tags),
        }

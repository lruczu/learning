from typing import List

import torch
from torch.utils.data.dataset import Dataset

from src.ner.config import Parameters
from src.ner.training.example import TestExample


class NERTestDataset(Dataset):
    def __init__(self, examples: List[TestExample], tokenizer, max_len: int = Parameters['MAX_LEN']):
        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.n_tags = 3

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        input_ids = self.tokenizer.encode(example.sentence)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len - 1]
            input_ids.append(102)

        padding_len = self.max_len - len(input_ids)
        token_type_ids = [0] * len(input_ids) + [0] * padding_len
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        input_ids = input_ids + [0] * padding_len

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'id': torch.tensor([example.id_]),
        }

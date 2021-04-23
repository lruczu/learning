from typing import List

import torch
from torch.utils.data.dataset import Dataset

from training.test_example import TestExample
from training.utils import prepare_inference_features


class QATestDataset(Dataset):
    def __init__(
        self,
        examples: List[TestExample],
        tokenizer,
        max_seq_length: str,
        doc_stride: int,
    ):
        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride

    def __getitem__(self, index):
        features = prepare_inference_features(
            self.examples[index].question,
            self.examples[index].context,
            self.tokenizer,
            self.max_seq_length,
            self.doc_stride
        )

        return {
            'input_ids': torch.tensor(features['input_ids'][0]),
            'attention_mask': torch.tensor(features['attention_mask'][0]),
            'token_type_ids': torch.tensor(features['token_type_ids'][0]),
            'offset_mapping_start': torch.tensor(features['offset_mapping_start'][0]),
            'offset_mapping_end': torch.tensor(features['offset_mapping_end'][0]),
            'example_index': torch.tensor(index),
        }

    def __len__(self):
        return len(self.examples)

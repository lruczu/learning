from typing import List

import torch
from torch.utils.data.dataset import Dataset

from training.example import Example


class QADataset(Dataset):
    def __init__(
        self,
        examples: List[Example],
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
        features = self.prepare_training_features(self.examples[index])

        return {
            'input_ids': torch.tensor(features['input_ids'][0]),
            'attention_mask': torch.tensor(features['attention_mask'][0]),
            'token_type_ids': torch.tensor(features['token_type_ids'][0]),
            'start_positions': torch.tensor(features['start_positions'][0]),
            'end_positions': torch.tensor(features['end_positions'][0]),
        }

    def __len__(self):
        return len(self.examples)

    def prepare_training_features(self, example: Example):
        tokenizer_output = self.tokenizer(
            example.question,
            example.context,
            truncation='only_second',
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        offset_mapping = tokenizer_output.pop("offset_mapping")

        tokenizer_output["start_positions"] = []
        tokenizer_output["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenizer_output["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            sequence_ids = tokenizer_output.sequence_ids(i)

            if example.is_impossible:
                tokenizer_output["start_positions"].append(cls_index)
                tokenizer_output["end_positions"].append(cls_index)
            else:
                start_char = example.answer_start_index
                end_char = start_char + len(example.answer)

                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenizer_output["start_positions"].append(cls_index)
                    tokenizer_output["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenizer_output["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenizer_output["end_positions"].append(token_end_index + 1)

        return tokenizer_output

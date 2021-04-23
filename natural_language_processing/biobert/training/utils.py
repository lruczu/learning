from typing import List, Tuple

import numpy as np
import torch


def get_batch(
    qas: List[Tuple[str, str]],
    tokenizer,
    max_seq_length: int,
    doc_stride: int,
):
    batch = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'offset_mapping_start': [],
        'offset_mapping_end': [],
    }
    for question, context in qas:
        features = prepare_inference_features(
            question,
            context,
            tokenizer,
            max_seq_length,
            doc_stride,
        )
        batch['input_ids'].append(features['input_ids'][0])
        batch['attention_mask'].append(features['attention_mask'][0])
        batch['token_type_ids'].append(features['token_type_ids'][0])
        batch['offset_mapping_start'].append(features['offset_mapping_start'][0])
        batch['offset_mapping_end'].append(features['offset_mapping_end'][0])

    batch['input_ids'] = torch.tensor(batch['input_ids'])
    batch['attention_mask'] = torch.tensor(batch['attention_mask'])
    batch['token_type_ids'] = torch.tensor(batch['token_type_ids'])
    batch['offset_mapping_start'] = np.array(batch['offset_mapping_start'])
    batch['offset_mapping_end'] = np.array(batch['offset_mapping_end'])

    return batch


def prepare_inference_features(
    question: str,
    context: str,
    tokenizer,
    max_seq_length: int,
    doc_stride: int,
):
    tokenizer_output = tokenizer(
        question,
        context,
        truncation='only_second',
        max_length=max_seq_length,
        stride=doc_stride,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,  # just to have nice dimensions
        padding="max_length"
    )
    tokenizer_output.pop('overflow_to_sample_mapping')

    tokenizer_output['offset_mapping_start'] = []
    tokenizer_output['offset_mapping_end'] = []

    for i, offsets in enumerate(tokenizer_output['offset_mapping']):
        sequence_ids = tokenizer_output.sequence_ids(i)

        tokenizer_output['offset_mapping_start'].append([
            o[0] if sequence_ids[ind] == 1 else -1
            for ind, o in enumerate(offsets)
        ])
        tokenizer_output['offset_mapping_end'].append([
            o[1] if sequence_ids[ind] == 1 else -1
            for ind, o in enumerate(offsets)
        ])

    tokenizer_output.pop('offset_mapping')

    return tokenizer_output

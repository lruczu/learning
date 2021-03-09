from typing import List

import torch

from common.qa import QA


@torch.no_grad()
def predict(model, tokenizer, qas: List[QA]):
    tensor_dict = tokenizer.tokenize(
        [qa.question for qa in qas],
        [qa.context for qa in qas],
    )
    output = model.predict(
        tensor_dict['input_ids'],
        tensor_dict['token_type_ids'],
        tensor_dict['attention_mask'],
    )

    output['start_logits']
    output['end_logits']


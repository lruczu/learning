from typing import Optional

import torch
from transformers import AutoModelForQuestionAnswering


# we might not need this class
class BioBERT:
    def __init__(
        self,
        checkpoint: Optional[str],
        pretrained_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        if (checkpoint is not None) + (pretrained_dir is not None) != 1:
            raise ValueError('You need to provide either checkpoint or pretrained_dir.')

        if checkpoint is not None:
            self.bert = AutoModelForQuestionAnswering.from_pretrained(
                checkpoint,
                cache_dir=cache_dir
            )
        else:
            self.bert = AutoModelForQuestionAnswering.from_pretrained(pretrained_dir)

    def predict(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
    ):
        return self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )

    def save(self, save_dir: str):
        self.bert.save_pretrained(save_dir)

from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel


from biobert.config import CHECKPOINT


class BioBERT(torch.Module.nn):
    """
    Based on paper:
    https://arxiv.org/abs/1909.08229
    """
    H = 768

    def __init__(self):
        super(BioBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(CHECKPOINT)
        self.start_v = torch.nn.Parameter(
            torch.normal(mean=torch(torch.zeros(BioBERT.H)), std=torch.tensor(0.2)),
            requires_grad=True)
        self.end_v = torch.nn.Parameter(
            torch.normal(mean=torch.zeros(BioBERT.H), std=torch.tensor(0.2)),
            requires_grad=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = output["last_hidden_state"]  # (batch_size, n_tokens, 768)
        _ = output["pooler_output"]  # (batch_size, 768)

        start_dot_product = torch.sum(last_hidden_state * self.start_v, axis=2)
        end_dot_product = torch.sum(last_hidden_state * self.end_v, axis=2)

        start_probs = F.softmax(start_dot_product) * token_type_ids
        end_probs = F.softmax(end_dot_product) * token_type_ids

        return start_probs, end_probs

    def predict(self, context: str, question: str):
        pass

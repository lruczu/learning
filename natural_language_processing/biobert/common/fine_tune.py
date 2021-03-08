from math import ceil
from typing import Optional

import datasets
import torch

from processing.biobert_tokenizer import BioBertTokenizer
from training.optimizer import get_optimizer_for_model
from training.scheduler import get_learning_rate_scheduler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_one_batch(
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output: torch.Tensor,
    model,
    optimizer,
    scheduler,
    gradient_clipping_value: Optional[float] = None
): # -> Tuple[T.Tensor, T.Tensor]:

    optimizer.zero_grad()

    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)

    start_logits, end_logits = model(input_ids, token_type_ids, attention_mask)
    loss = criterion(output, mm)

    # backward
    loss.backward()

    if gradient_clipping_value is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clipping_value)

    optimizer.step()
    scheduler.step()

    return loss


def fine_tune(
    dataset: datasets.dataset_dict.DatasetDict,
    model,
    tokenizer: BioBertTokenizer,
    n_epochs: int = 3,
    batch_size: int = 32,
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    optimizer = get_optimizer_for_model(model)
    scheduler = get_learning_rate_scheduler(optimizer,
                                            n_steps=ceil(dataset.num_rows / batch_size) * n_epochs)

    for epoch in range(n_epochs):
        for i, batch in enumerate(dataloader):  # batch of type dict
            tensor_dict = tokenizer.tokenize(batch['question'], batch['context'])

            train_one_batch(
                input_ids=tensor_dict['input_ids'],
                token_type_ids=tensor_dict['token_type_ids'],
                attention_mask=tensor_dict['attention_mask'],
                output=None,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

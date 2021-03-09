from math import ceil
import random
import os
from typing import Optional

import datasets
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from processing.biobert_tokenizer import BioBertTokenizer
from training.optimizer import get_optimizer_for_model
from training.scheduler import get_learning_rate_scheduler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def make_deterministic(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_one_batch(
    epoch: int,
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_positions: torch.Tensor,
    end_positions: torch.Tensor,
    model,
    optimizer,
    scheduler,
    summary_writer,
    gradient_clipping_value: Optional[float] = None
) -> torch.Tensor:

    optimizer.zero_grad()

    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)

    output = model(input_ids, token_type_ids, attention_mask, start_positions, end_positions)
    loss = output['loss']
    summary_writer.add_scalar(f'Epoch_{epoch}/train', loss.item())

    # backward
    loss.backward()

    if gradient_clipping_value is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clipping_value)

    optimizer.step()
    scheduler.step()

    return loss


def fine_tune(
    dataset: datasets.Dataset,
    model,
    tokenizer: BioBertTokenizer,
    log_dir: str,
    n_epochs: int = 3,
    batch_size: int = 32,
    seed: int = 123,
):
    writer = SummaryWriter(log_dir=log_dir)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    optimizer = get_optimizer_for_model(model)
    scheduler = get_learning_rate_scheduler(optimizer,
                                            n_steps=ceil(dataset.num_rows / batch_size) * n_epochs)

    make_deterministic(seed)

    for epoch in range(n_epochs):
        for i, batch in enumerate(dataloader):  # batch of type dict
            tensor_dict = tokenizer.tokenize(batch['question'], batch['context'])

            train_one_batch(
                epoch=epoch+1,
                input_ids=tensor_dict['input_ids'],
                token_type_ids=tensor_dict['token_type_ids'],
                attention_mask=tensor_dict['attention_mask'],
                start_positions=torch.tensor(batch['start_index']),
                end_positions=torch.tensor(batch['end_index']),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                summary_writer=writer,
            )

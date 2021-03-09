from math import ceil
import random
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import (
    BATCH_SIZE,
    LOG_DIR,
    LR,
    MODEL_CHECKPOINT_SAVE,
    N_EPOCHS,
    OUTPUT_DIRECTORY,
    TRAIN_DATA_PATH,
    WEIGHT_DECAY,
)
from model.biobert2 import BioBERT
from processing.biobert_tokenizer import BioBertTokenizer
from processing.io import load
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
    step: int,
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

    model.train()
    output = model(input_ids, token_type_ids, attention_mask, start_positions, end_positions)
    loss = output['loss']

    # if eval
    # model.eval()

    # logging
    summary_writer.add_scalar(f'Epoch_{epoch}/train', loss.item())
    summary_writer.add_scalar(f'Epoch_{epoch}/lr', scheduler.get_last_lr()[0])

    # backward
    loss.backward()

    if gradient_clipping_value is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clipping_value)

    optimizer.step()
    scheduler.step()

    if step % MODEL_CHECKPOINT_SAVE == 0:
        model.save(OUTPUT_DIRECTORY)

    return loss


def train(model, tokenizer, seed: int = 123):
    writer = SummaryWriter(log_dir=LOG_DIR)
    dataset = load(TRAIN_DATA_PATH)
    dataset.shuffle(seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = get_optimizer_for_model(model, WEIGHT_DECAY, LR)
    scheduler = get_learning_rate_scheduler(optimizer,
                                            n_steps=ceil(dataset.num_rows / BATCH_SIZE) * N_EPOCHS)

    make_deterministic(seed)

    step = 0
    for epoch in range(N_EPOCHS):
        for i, batch in enumerate(dataloader):  # batch of type dict
            tensor_dict = tokenizer.tokenize(batch['question'], batch['context'])

            train_one_batch(
                epoch=epoch+1,
                step=step+1,
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
            step += 1


if __name__ == '__main__':
    # TODO: refactor it
    biobert = BioBERT(MODEL_CHECKPOINT)
    biobert_tokenizer = BioBertTokenizer(MODEL_CHECKPOINT)
    train(biobert, biobert_tokenizer)

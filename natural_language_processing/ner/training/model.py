from typing import Optional
import os

import pytorch_lightning as pl
import torch
from transformers import (
    AdamW,
    BertModel,
    get_polynomial_decay_schedule_with_warmup,
)


class LastLayer(torch.nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(768, self.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def get_dict_repr(self):
        d = {}
        for index, module in enumerate(self.named_modules()):
            if index == 0:
                continue
            d[str(index)] = str(module[1])
        return d


class NERModel(pl.LightningModule):
    NUM_LABELS = 3

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        n_steps: int = 10000,
        weight_decay: float = 0.,
        lr: float = 3e-5,
        warm_up_prop: float = 0.,
        run=None,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(checkpoint) if checkpoint else None
        self.last_layer = LastLayer(NERModel.NUM_LABELS)

        self.n_steps = n_steps
        self.weight_decay = weight_decay
        self.lr = lr
        self.warm_up_prop = warm_up_prop
        self.run = run

        self.cel = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        bert_output = self.bert(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids'],
        )
        output = self.last_layer(bert_output['last_hidden_state'])
        return output

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        if self.run:
            self.run['train/loss'].log(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        if self.run:
            self.run['validation/loss'].log(loss)

    def shared_step(self, batch):
        output = self.forward(batch)
        return self.compute_loss(batch, output)

    def compute_loss(self, batch, output):
        where_active = batch['attention_mask'].view(-1) == 1
        active_logits = output.view(-1, NERModel.NUM_LABELS)
        active_labels = torch.where(
            where_active,
            batch['tags'].view(-1),
            torch.tensor(self.cel.ignore_index).type_as(batch['tags'])
        )
        return self.cel(active_logits, active_labels)

    def get_optimizer_for_model(self, weight_decay: float, lr: float):
        def apply_decay(name: str) -> bool:
            if 'bias' in name:
                return False
            elif 'LayerNorm.weight' in name:
                return False
            return True

        params = [
            {'params':
                 [tensor for name, tensor in self.bert.named_parameters() if apply_decay(name)],
             'weight_decay': weight_decay,
             },
            {'params':
                 [tensor for name, tensor in self.bert.named_parameters() if not apply_decay(name)],
             'weight_decay': 0.0,
             }
        ]

        return AdamW(
            params=params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-6,
        )

    def get_learning_rate_scheduler(self, optimizer, n_steps: int, warm_up_prop: float):
        return get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(warm_up_prop * n_steps),
            num_training_steps=n_steps,
        )

    def configure_optimizers(self):
        optimizer = self.get_optimizer_for_model(self.weight_decay, self.lr)
        scheduler = self.get_learning_rate_scheduler(optimizer, self.n_steps, self.warm_up_prop)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def save(self, checkpoint: str):
        self.bert.save_pretrained(os.path.join(checkpoint, 'bert.pt'))
        torch.save(self.last_layer, os.path.join(checkpoint, 'last_layer.pt'))

    def load(self, checkpoint: str):
        self.bert = BertModel.from_pretrained(os.path.join(checkpoint, 'bert.pt'))
        self.last_layer = torch.load(os.path.join(checkpoint, 'last_layer.pt'))

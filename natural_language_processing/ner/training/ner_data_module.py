from math import ceil
from typing import List

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.ner.training.example import Example
from src.ner.training.ner_dataset import NERDataset


class NERDataModule(pl.LightningDataModule):
    def __init__(
        self,
        examples: List[Example],
        tokenizer,
        batch_size: int,
        val_prop: float,
        max_len: int,
    ):
        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.val_prop = val_prop
        self.max_len = max_len

        self.n_samples = len(self.examples)

    def prepare_data(self):
        ...

    def setup(self, stage=None):
        self.train, self.valid = train_test_split(self.examples, test_size=self.val_prop)
        self.train = NERDataset(self.train, self.tokenizer, self.max_len)
        self.valid = NERDataset(self.valid, self.tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def number_of_steps_per_epoch(self):
        n_samples = int((1 - self.val_prop) * self.n_samples)
        return ceil(n_samples / self.batch_size)

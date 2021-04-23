from math import ceil
from typing import List, Union

import pytorch_lightning as pl
from transformers import (
    AutoTokenizer,
)
from torch.utils.data import DataLoader

from config import CACHE_DIR
from training.example import Example
from training.io import read_number_of_samples, load_examples
from training.qa_dataset import QADataset


class QADataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_name_or_path,
        train_path: Union[str, List[str]],
        valid_path: Union[str, List[str]],
        batch_size: int,
        max_seq_length: int,
        doc_stride: int,
        max_query_length: int,
        cache_dir: str = CACHE_DIR,
    ):
        super().__init__()

        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path,
                                                       use_fast=True, cache_dir=cache_dir)

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True)

    def setup(self, stage=None):
        train_examples: List[Example] = load_examples(self.train_path)
        valid_examples: List[Example] = load_examples(self.valid_path)

        self.train = QADataset(train_examples, self.tokenizer, self.max_seq_length, self.doc_stride)
        self.valid = QADataset(valid_examples, self.tokenizer, self.max_seq_length, self.doc_stride)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def number_of_steps_per_epoch(self):
        n_samples = read_number_of_samples(self.train_path)
        return ceil(n_samples / self.batch_size)

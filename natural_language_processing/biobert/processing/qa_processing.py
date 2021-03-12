import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from transformers import AutoTokenizer


class QAProcessing(pl.LightningDataModule):
    def __init__(self, batch_size: int):

        super().__init__()

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self):
        self.train_data = None
        self.valid_data = None

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size)

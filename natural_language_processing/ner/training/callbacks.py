from functools import reduce
from typing import List

from neptune.new.types import File
import pandas as pd
from pytorch_lightning.callbacks import Callback, LearningRateMonitor

from src.ner.common.structures import Document
from src.ner.inference.inference_utils import SentenceExtractor
from src.ner.inference.predict import get_datasets_from_sentence


class LearningRateLogging(LearningRateMonitor):
    def __init__(self, run=None):
        super().__init__()
        self.run = run

    def on_train_batch_start(self, trainer, *args, **kwargs):
        latest_stat = self._extract_stats(trainer, 'step')

        if self.run:
            for key, value in latest_stat.items():
                self.run['train/' + key].log(value)


class DisplayExamples(Callback):
    def __init__(self, sentences: List[str], tokenizer, run=None):
        super().__init__()
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.run = run

    def on_epoch_end(self, trainer, pl_module):
        if self.run is None:
            return

        all_datasets = []
        for sentence in self.sentences:
            datasets = get_datasets_from_sentence(pl_module, self.tokenizer, sentence)
            all_datasets.append('|'.join(datasets))

        df = pd.DataFrame({
            'sentence': self.sentences,
            'datasets': all_datasets
        })

        self.run[f'display_{pl_module.current_epoch}'].upload(File.as_html(df))


class SubmissionFiles(Callback):
    def __init__(self, docs: List[Document], tokenizer, run=None):
        super().__init__()
        self.docs = docs
        self.tokenizer = tokenizer
        self.run = run

    def on_epoch_end(self, trainer, pl_module):
        if self.run is None:
            return

        all_datasets = []
        ids = []
        for doc in self.docs:
            ids.append(doc.document_id)
            sentences = SentenceExtractor.get_all_valid_sentences(doc)
            datasets = []
            for sentence in sentences:
                datasets.append(
                    get_datasets_from_sentence(pl_module, self.tokenizer, sentence)
                )

            datasets = reduce(lambda a, b: a + b, datasets)
            datasets = list(set(datasets))
            all_datasets.append('|'.join(datasets))
        df = pd.DataFrame({
            'document_id': ids,
            'datasets': all_datasets
        })
        self.run[f'submission_{pl_module.current_epoch}'].upload(File.as_html(df))

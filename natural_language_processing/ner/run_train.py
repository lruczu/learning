import random
import os

import neptune.new as neptune
import numpy as np
import pytorch_lightning as pl
import torch as T
from transformers import BertTokenizer

from src.ner.common.structures import Document
from src.ner.config import CHECKPOINT, EXAMPLES_TO_DISPLAY, Parameters
from src.ner.training.callbacks import DisplayExamples, LearningRateLogging, SubmissionFiles
from src.ner.training.model import NERModel
from src.ner.training.ner_data_module import NERDataModule
from src.ner.training.train_utils import read_training_set


def make_deterministic(seed: int = 123) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.backends.cudnn.deterministic = True


def main():
    make_deterministic()
    run = neptune.init(project='Netguru/Coleridge')
    run['parameters'] = Parameters

    tokenizer = BertTokenizer.from_pretrained(CHECKPOINT)
    examples = read_training_set(Parameters['TRAINING_PATH'])
    data_module = NERDataModule(
        examples,
        tokenizer,
        Parameters['TRAIN_BATCH_SIZE'],
        Parameters['VAL_PROP'],
        Parameters['MAX_LEN'],
    )
    model = NERModel(
        CHECKPOINT,
        n_steps=Parameters['N_EPOCHS'] * data_module.number_of_steps_per_epoch(),
        weight_decay=Parameters['WEIGHT_DECAY'],
        lr=Parameters['lr'],
        warm_up_prop=Parameters['WARM_UP_PROP'],
        run=run,
    )
    run['last_layer_architecture'] = model.last_layer.get_dict_repr()

    docs = [
        Document.read_from_path(path) for path in Parameters['submission_paths']
    ]
    trainer = pl.Trainer(
        max_epochs=Parameters['N_EPOCHS'],
        gpus=1,
        callbacks=[LearningRateLogging(run),
                   DisplayExamples(EXAMPLES_TO_DISPLAY, tokenizer, run),
                   SubmissionFiles(docs, tokenizer, run)
                   ],
    )

    trainer.fit(model, data_module)

    run.stop()


if __name__ == '__main__':
    main()

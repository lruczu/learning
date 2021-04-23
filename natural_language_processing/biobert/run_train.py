from argparse import ArgumentParser
import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from config import PreprocessingConfig, TrainingConfig
from training.biobert import BioBERT
from training.callbacks import SaveCallback
from training.qa_data_module import QADataModule


def main(args):
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(
        TrainingConfig.EXPERIMENT_NAME, 'logs'
    ))

    qa_data_module = QADataModule(
        tokenizer_name_or_path=TrainingConfig.TOKENIZER_CHECKPOINT,
        train_path=TrainingConfig.TRAIN_DATA_PATH,
        valid_path=TrainingConfig.VALID_DATA_PATH,
        batch_size=TrainingConfig.BATCH_SIZE,
        max_seq_length=PreprocessingConfig.MAX_LENGTH,
        doc_stride=PreprocessingConfig.DOC_STRIDE,
        max_query_length=PreprocessingConfig.MAX_QUERY_LENGTH,
    )

    biobert = BioBERT(
        model_name_or_path=TrainingConfig.MODEL_CHECKPOINT,
        n_steps=TrainingConfig.N_EPOCHS * qa_data_module.number_of_steps_per_epoch(),
        lr=TrainingConfig.LR,
        weight_decay=TrainingConfig.WEIGHT_DECAY,
        warm_up_prop=TrainingConfig.WARM_UP_PROP,
        model_save_dir=TrainingConfig.EXPERIMENT_NAME
    )

    trainer = pl.Trainer(
        max_epochs=TrainingConfig.N_EPOCHS,
        gpus=args.gpus,
        callbacks=[SaveCallback(TrainingConfig.EXPERIMENT_NAME)],
        logger=tb_logger,
    )

    trainer.fit(biobert, qa_data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    args = parser.parse_args()

    main(args)

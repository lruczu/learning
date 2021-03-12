import pytorch_lightning as pl
from transformers import AutoModelForQuestionAnswering

from training.optimizer import get_optimizer_for_model
from training.scheduler import get_learning_rate_scheduler


class BioBERT(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        n_steps: int,
        experiment_name: str,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.n_steps = n_steps
        self.experiment_name = experiment_name

        self.bert = AutoModelForQuestionAnswering.from_pretrained(
            self.model_name_or_path,
            cache_dir=experiment_name
        )

    def training_step(self, batch, batch_index):
        output = self.bert(
            input_ids=batch['batch'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask'],
            start_positions=batch['start_positions'],
            end_positions=batch['end_positions'],
        )
        loss = output['loss']
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = get_optimizer_for_model(self, 0.01, 5e-5)
        scheduler = get_learning_rate_scheduler(optimizer, self.n_steps, 0.1)

        return [optimizer], [scheduler]

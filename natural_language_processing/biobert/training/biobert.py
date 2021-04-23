import pytorch_lightning as pl
from transformers import AutoModelForQuestionAnswering

from config import CACHE_DIR
from training.optimizer import get_optimizer_for_model
from training.scheduler import get_learning_rate_scheduler


class BioBERT(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        n_steps: int,
        lr: float,
        weight_decay: float,
        warm_up_prop: float,
        model_save_dir: str,
        cache_dir: str = CACHE_DIR
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.n_steps = n_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.warm_up_prop = warm_up_prop
        self.model_save_dir = model_save_dir
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_name_or_path,
            cache_dir=cache_dir
        )

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val_loss', loss)

    def shared_step(self, batch):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            start_positions=batch['start_positions'],
            end_positions=batch['end_positions'],
        )
        return output['loss']

    def configure_optimizers(self):
        optimizer = get_optimizer_for_model(self, self.weight_decay, self.lr)
        scheduler = get_learning_rate_scheduler(optimizer, self.n_steps, self.warm_up_prop)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

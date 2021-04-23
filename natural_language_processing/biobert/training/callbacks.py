import os

from pytorch_lightning.callbacks.base import Callback


class SaveCallback(Callback):
    def __init__(self, model_save_dir: str):
        self.model_save_dir = model_save_dir

    def on_epoch_end(self, trainer, pl_module):
        pl_module.model.save_pretrained(
            os.path.join(self.model_save_dir, str(pl_module.current_epoch)))

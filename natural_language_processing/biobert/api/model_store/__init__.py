from pydantic import BaseSettings

from .store import ModelStore
from .aws_store import AWSModelStore


class ModelStoreSettings(BaseSettings):
    bucket_name: str


def get_model_store(settings: ModelStoreSettings = None) -> ModelStore:
    settings = settings or ModelStoreSettings()
    return AWSModelStore(settings.bucket_name)

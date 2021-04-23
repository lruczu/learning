import os

from pydantic import BaseSettings

from api.model_store import ModelStore
import boto3


class AWSStoreSettings(BaseSettings):
    s3_aws_access_key_id: str = ""
    s3_aws_secret_access_key: str = ""


class AWSModelStore(ModelStore):
    def __init__(self, bucket_name, resource=None):
        self.bucket_name = bucket_name
        self.resource = resource or self.construct_from_settings(AWSStoreSettings())
        self.bucket = self.resource.Bucket(self.bucket_name)

    @classmethod
    def construct_from_settings(cls, settings: AWSStoreSettings):
        return boto3.resource(
            "s3",
            aws_access_key_id=settings.s3_aws_access_key_id,
            aws_secret_access_key=settings.s3_aws_secret_access_key,
        )

    def download_model(self, model_name):
        objects = list(
            self.bucket.objects.filter(Prefix=self.get_model_path(model_name))
        )
        if not objects:
            raise ValueError("Not found")

        os.makedirs(self.get_model_path(model_name), exist_ok=True)

        for obj in objects:
            if obj.key.endswith("/"):
                continue
            self.bucket.download_file(obj.key, obj.key)

    def model_exists(self, model_name):
        return os.path.isdir(self.get_model_path(model_name))

    def get_model_path(self, model_name):
        return f"models/{model_name}"

    def get_available_models(self):
        return list(
            set(
                name
                for obj in self.bucket.objects.filter(Prefix="models/")
                if (name := obj.key.split("/")[1])
            )
        )

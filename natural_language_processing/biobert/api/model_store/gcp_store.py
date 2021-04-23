import os

from google.cloud import storage

from api.model_store.store import ModelStore


class GCPModelStore(ModelStore):
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket: storage.Bucket = self.client.bucket(bucket_name)

    def get_model_path(self, model_name):
        return f"models/{model_name}"

    def download_model(self, model_name):
        blobs = list(self.client.list_blobs(self.bucket_name, prefix=model_name))
        if not blobs:
            raise ValueError("Not found")

        os.makedirs(self.get_model_path(model_name), exist_ok=True)

        for blob in blobs:
            blob.download_to_filename(f"models/{blob.name}")

    def model_exists(self, model_name):
        return os.path.isdir(self.get_model_path(model_name))

    def get_available_models(self):
        return list(
            set(
                blob.name.split("/")[0]
                for blob in self.client.list_blobs(self.bucket_name)
            )
        )

from abc import ABC, abstractmethod


class ModelStore(ABC):
    @abstractmethod
    def download_model(self, model_name):
        pass

    @abstractmethod
    def model_exists(self, model_name):
        pass

    @abstractmethod
    def get_model_path(self, model_name):
        pass

    @abstractmethod
    def get_available_models(self):
        pass

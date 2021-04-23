from abc import abstractmethod, ABC
from fastapi.logger import logger
from pydantic import BaseSettings

from api.qa import QA, SearchResult
from inference import predict as biobert_predict, fast_predict as biobert_fast_predict
from inference.loading import load_model


class InferenceSettings(BaseSettings):
    batch_size: int = 16
    use_cuda: bool = False
    fast_predict: bool = True

    class Config:
        env_prefix = "inference_"


class PredictorInterface(ABC):
    def __init__(self, config: InferenceSettings = None):
        self.config = config or InferenceSettings()

    @abstractmethod
    def predict(self, qa: QA, model):
        pass

    @abstractmethod
    def load_model(self, model_path):
        pass

    def get_context(self, result: SearchResult):
        return f"{result.title} {result.sentence}"

    def get_contexts(self, qa: QA):
        return [self.get_context(r) for r in qa.results]

    @abstractmethod
    def get_default_model(self):
        pass


class Predictor(PredictorInterface):
    def predict(self, qa: QA, model):
        logger.debug('Using slow Predictor')
        contexts = [self.get_context(r) for r in qa.results]
        return [
            biobert_predict.predict(qa.question, context, model) for context in contexts
        ]

    def load_model(self, model_path):
        return load_model(model_path)

    def get_default_model(self):
        return biobert_predict.model_to_pipeline()


class FastPredictor(PredictorInterface):
    def predict(self, qa: QA, model):
        logger.debug('Using FastPredictor')
        qas = [(qa.question, self.get_context(r)) for r in qa.results]
        results = biobert_fast_predict.fast_predict(
            qas, model, batch_size=self.config.batch_size, use_cuda=self.config.use_cuda
        )
        return [(answer, float(score)) for answer, score in results]

    def load_model(self, model_path):
        model = load_model(model_path)
        if self.config.use_cuda:
            model.to("cuda")
        model.eval()
        return model

    def get_default_model(self):
        return biobert_fast_predict.get_default_model(use_cuda=self.config.use_cuda)


def get_predictor(settings: InferenceSettings = None) -> PredictorInterface:
    settings = settings or InferenceSettings()
    if settings.fast_predict:
        return FastPredictor(settings)
    return Predictor(settings)

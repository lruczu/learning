import logging
import threading

from fastapi import FastAPI, Depends
from fastapi.logger import logger
from functools import lru_cache
from pydantic import BaseSettings
from typing import Optional

from api.model_store import get_model_store
from api.predictor import get_predictor
from api.qa import QA, SearchResult
from api.utils import locked

app = FastAPI()


class ApiSettings(BaseSettings):
    debug: bool = True


class CurrentModel:
    def __init__(self, model=None):
        self.name = "default"
        self.__model = model

    @property
    def model(self):
        if not self.__model:
            self.__model = get_predictor().get_default_model()
        return self.__model

    @model.setter
    def model(self, new_model):
        self.__model = new_model


@lru_cache
def get_current_model():
    predictor = get_predictor()
    return CurrentModel(model=predictor.get_default_model())


@lru_cache
def get_model_lock() -> threading.Lock:
    return threading.Lock()


def get_context(result: SearchResult):
    return f"{result.title} {result.sentence}"


def get_default_predictor():
    return get_predictor()


@app.post("/v1/qa")
def get_answer(qa: QA, current_model=Depends(get_current_model), predictor=Depends(get_default_predictor)):
    contexts = predictor.get_contexts(qa)
    results = predictor.predict(qa, current_model.model)
    return [
        {"answer": answer, "score": score, "context": context}
        for (answer, score), context in zip(results, contexts)
    ]


@app.put("/v1/current-model")
def swap_model(
    model_name: str,
    overwrite: Optional[bool] = False,
    current_model=Depends(get_current_model),
    store=Depends(get_model_store),
    predictor=Depends(get_predictor),
    model_lock: threading.Lock = Depends(get_model_lock)
):
    with locked(model_lock) as is_locked:
        if not is_locked:
            logger.debug(f'Lock {model_lock} is locked for {model_name}. Skipping')
            return {'status': 'locked'}

        logger.debug(f'Lock {model_lock} acquired for model {model_name}.')
        if current_model.name == model_name:
            return {'status': 'skipped'}

        if not store.model_exists(model_name) or overwrite:
            store.download_model(model_name)
        model_path = store.get_model_path(model_name)

        current_model.name = model_name
        current_model.model = predictor.load_model(model_path)

        return {'status': 'ok'}


@app.get("/v1/current-model")
async def get_model(current_model=Depends(get_current_model), store=Depends(get_model_store)):
    return {"current": current_model.name, "available": store.get_available_models()}


@app.on_event("startup")
async def init_logger():
    settings = ApiSettings()
    if settings.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@app.on_event("startup")
async def initialize_model():
    current_model = get_current_model()
    logger.debug(f'Current model: {current_model.name}')

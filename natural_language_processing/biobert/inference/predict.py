from functools import lru_cache

from typing import Tuple

from transformers import pipeline, Pipeline

from config import Inference, PreprocessingConfig
from inference.loading import load_model, load_tokenizer


@lru_cache
def get_default_model():
    return load_model(Inference.MODEL_PATH)


@lru_cache
def get_default_tokenizer():
    return load_tokenizer(Inference.TOKENIZER_PATH)


def model_to_pipeline(model=None, tokenizer=None):
    model = model or get_default_model()
    tokenizer = tokenizer or get_default_tokenizer()
    return pipeline(
        'question-answering',
        model,
        tokenizer=tokenizer,
        use_fast=True
    )


@lru_cache
def get_default_nlp():
    return model_to_pipeline(get_default_model(), get_default_tokenizer())


def predict(question: str, context: str, nlp: Pipeline = None) -> Tuple[str, float]:
    """
    Returns:
        (answer, score)
    """
    nlp_ = nlp or get_default_nlp()
    output = nlp_(
        question,
        context,
        max_seq_len=PreprocessingConfig.MAX_LENGTH,
        max_question_len=PreprocessingConfig.MAX_QUERY_LENGTH,
        doc_stride=PreprocessingConfig.DOC_STRIDE,
        handle_impossible_answer=True,
    )
    return output['answer'], output['score']

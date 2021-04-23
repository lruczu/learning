from functools import lru_cache

from typing import List, Tuple

import torch

from config import Inference, PreprocessingConfig
from inference.loading import load_model, load_tokenizer
from inference.utils import get_best_answer
from training.utils import get_batch


@lru_cache
def get_default_model(use_cuda):
    model = load_model(Inference.MODEL_PATH)

    if use_cuda:
        model = model.to('cuda')

    model.eval()
    return model


@lru_cache
def get_default_tokenizer():
    return load_tokenizer(Inference.TOKENIZER_PATH)


def fast_predict(qas: List[Tuple[str, str]], model=None, tokenizer=None, use_cuda=False, batch_size=16) -> List[Tuple[str, float]]:
    model = model or get_default_model(use_cuda=use_cuda)
    tokenizer = tokenizer or get_default_tokenizer()
    n_requests = len(qas)

    return [
        result
        for i in range(0, n_requests, batch_size)
        for result in fast_predict_(qas[i:i+batch_size], model=model, tokenizer=tokenizer, use_cuda=use_cuda)
    ]


def fast_predict_(qas: List[Tuple[str, str]], model=None, tokenizer=None, use_cuda=False) -> List[Tuple[str, float]]:
    """
    Args:
        qas: (question, context)
        model
        tokenizer
        use_cuda

    Returns:
        (answer, score)
    """
    model = model or get_default_model(use_cuda=use_cuda)
    tokenizer = tokenizer or get_default_tokenizer()

    n_samples = len(qas)
    results = []
    with torch.no_grad():
        batch = get_batch(qas,
                          tokenizer,
                          PreprocessingConfig.MAX_LENGTH,
                          PreprocessingConfig.DOC_STRIDE)
        output = model(
            input_ids=batch['input_ids'].to('cuda') if use_cuda else batch['input_ids'],
            attention_mask=batch['attention_mask'].to('cuda') if use_cuda else batch['attention_mask'],
            token_type_ids=batch['token_type_ids'].to('cuda') if use_cuda else batch['token_type_ids'],
        )
        for i in range(n_samples):
            answer = get_best_answer(
                start_logits=output['start_logits'][i].cpu().numpy()
                if use_cuda else output['start_logits'][i].detach().numpy(),
                end_logits=output['end_logits'][i].cpu().numpy()
                if use_cuda else output['end_logits'][i].detach().numpy(),
                offset_mapping_start=batch['offset_mapping_start'][i],
                offset_mapping_end=batch['offset_mapping_end'][i],
                context=qas[i][1],
            )
            results.append((answer['answer'], answer['score']))

    return results

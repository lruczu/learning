from typing import Optional

from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def load_model(model_path: str):
    return AutoModelForQuestionAnswering.from_pretrained(model_path)


def load_tokenizer(tokenizer_path: str, cache_dir: Optional[str] = None):
    return AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_dir, use_fast=True)
